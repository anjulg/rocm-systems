"""Simulator plugin protocol for mirages.

Every simulator that mirages can drive must satisfy the :class:`Simulator`
:pep:`544` ``Protocol``.  The protocol is intentionally defined as a
structural type so that third-party packages can implement it without
importing or subclassing anything from mirages directly.

Data types
----------
All data types exchanged between the daemon and a simulator are defined
as FlatBuffer tables in ``schema/simulator.fbs`` (and ``schema/common.fbs``).
The Python-side representations are thin typed wrappers — see
``mirages.types`` for the full list.

Registration
------------
Simulators register themselves as Python entry-points under the group
``mirages.simulators``::

    # pyproject.toml
    [project.entry-points."mirages.simulators"]
    rocjitsu = "mirages.simulators.rocjitsu:RocjitsuSimulator"

The daemon loads all registered entry-points at startup.

Lifecycle
---------
1. The daemon calls :meth:`info` once at startup to discover the
   simulator's supported GPUs, modes, and capabilities.
2. When a user creates a session the daemon calls :meth:`create_session`
   which returns a :class:`ContainerDef` describing exactly how to launch
   the OCI container for that session.
3. Before every run inside a session the daemon calls
   :meth:`get_exec_run_def` so the simulator can inject environment
   variables, wrapper commands, or rewrite the command line.
4. The daemon periodically polls :meth:`session_health` and
   :meth:`session_perf` to surface status in the dashboard.
5. When a session is torn down the daemon calls :meth:`delete_session`.

Custom GPUs
-----------
Simulators that advertise ``supports_custom_gpus = True`` in their
:class:`SimulatorInfo` must implement :meth:`set_custom_gpu`.  This
lets users design GPU topologies (e.g. number of XCDs, shader engines,
CUs, wavefront slots) through the dashboard's graphical editor.

Example (rocjitsu)::

    sim = RocjitsuSimulator()
    info = sim.info()
    assert "MI300X" in [g.name for g in info.supported_gpus]

    session = SessionDef(name="bench-1", profile="mi300x-func")
    profile = ProfileDef(
        name="mi300x-func",
        simulator="rocjitsu",
        mode=SimulatorMode.Functional,
        gpu="MI300X",
        num_gpus=8,
        num_nodes=1,
    )
    container = sim.create_session(session, profile)
    # container.env includes LD_PRELOAD=librocjitsu.so, etc.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mirages.types import (
    ContainerDef,
    CustomGpuDef,
    ExecDef,
    GpuDef,
    ProfileDef,
    RunDef,
    SessionDef,
    SessionHealth,
    SessionPerf,
    SimulatorInfo,
)


@runtime_checkable
class Simulator(Protocol):
    """Protocol that every mirages simulator plugin must satisfy.

    All methods receive and return FlatBuffer-derived types defined in
    ``mirages.types``.  Simulators are expected to be **stateful** — the
    daemon instantiates each simulator once and keeps it alive for the
    lifetime of the daemon process.
    """

    # ------------------------------------------------------------------
    #  Identity
    # ------------------------------------------------------------------

    def info(self) -> SimulatorInfo:
        """Return metadata about this simulator.

        Called once at daemon startup.  The returned :class:`SimulatorInfo`
        tells the daemon which GPUs and modes are supported, whether
        custom GPU design is available, and provides display metadata for
        the dashboard.

        Returns
        -------
        SimulatorInfo
            Simulator identity, supported GPUs, supported modes, and
            feature flags.
        """
        ...

    # ------------------------------------------------------------------
    #  GPU discovery & customisation
    # ------------------------------------------------------------------

    def supported_gpus(self) -> list[GpuDef]:
        """Return the list of GPU models this simulator can emulate.

        This is a convenience accessor — the same list is embedded in the
        :class:`SimulatorInfo` returned by :meth:`info`.  The daemon may
        call this independently when it needs to refresh GPU availability
        without re-querying the full info payload.

        Returns
        -------
        list[GpuDef]
            One entry per supported GPU model.  Each ``GpuDef.name`` must
            be unique within this simulator.
        """
        ...

    def supports_custom_gpus(self) -> bool:
        """Whether this simulator supports user-defined GPU topologies.

        When ``True``, the dashboard shows a *Design Custom GPU* button
        and the daemon will call :meth:`set_custom_gpu` when the user
        submits a custom topology.

        Returns
        -------
        bool
        """
        ...

    def set_custom_gpu(self, gpu: CustomGpuDef) -> GpuDef:
        """Register a user-designed custom GPU topology.

        The simulator validates ``gpu.topology_json`` against its own
        schema and, if valid, returns a :class:`GpuDef` that can be
        referenced by name in future profiles.

        Parameters
        ----------
        gpu : CustomGpuDef
            The custom GPU definition including a name and a JSON
            topology string.  For rocjitsu this is the ``topology``
            subtree of a ``simulation_config.fbs`` document.

        Returns
        -------
        GpuDef
            A newly created GPU definition.  The ``name`` field matches
            ``gpu.name`` and can be used in ``ProfileDef.gpu``.

        Raises
        ------
        ValueError
            If the topology JSON is invalid or fails schema validation.
        NotImplementedError
            If :meth:`supports_custom_gpus` returns ``False``.
        """
        ...

    # ------------------------------------------------------------------
    #  Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self, session: SessionDef, profile: ProfileDef
    ) -> ContainerDef:
        """Create a new simulation session and return its container spec.

        The daemon calls this when the user creates a session.  The
        simulator should:

        1. Generate any configuration files needed inside the container
           (e.g. JSON simulation configs, FlatBuffers schemas).
        2. Determine which host libraries need to be bind-mounted.
        3. Compute the environment variables required for interposition
           (e.g. ``LD_PRELOAD``).
        4. Return a :class:`ContainerDef` that the daemon can hand
           directly to an OCI runtime.

        The simulator may also allocate internal bookkeeping state keyed
        by ``session.name``.

        Parameters
        ----------
        session : SessionDef
            The session to create, including its unique name and the
            container image.
        profile : ProfileDef
            The resolved profile, providing the GPU model, simulation
            mode, and cluster shape.

        Returns
        -------
        ContainerDef
            Complete container specification including image, env vars,
            bind-mounts, and injected files.
        """
        ...

    def delete_session(self, session_id: str) -> None:
        """Tear down a simulation session and release resources.

        Called when the user deletes a session or when the daemon shuts
        down.  The simulator should clean up any internal state, temp
        files, or resources associated with ``session_id``.

        Parameters
        ----------
        session_id : str
            The unique name of the session to delete.

        Raises
        ------
        KeyError
            If no session with this ID exists.
        """
        ...

    # ------------------------------------------------------------------
    #  Session observability
    # ------------------------------------------------------------------

    def session_health(self, session_id: str) -> SessionHealth:
        """Query the health of a running session.

        The daemon polls this periodically.  Simulators should return
        ``HealthStatus.Healthy`` when the simulation engine inside the
        container is responsive and processing events.

        Parameters
        ----------
        session_id : str
            The session to query.

        Returns
        -------
        SessionHealth
            Current health status, uptime, and optional error message /
            diagnostics.

        Raises
        ------
        KeyError
            If no session with this ID exists.
        """
        ...

    def session_perf(self, session_id: str) -> SessionPerf:
        """Query performance counters for a running session.

        Returns simulation-time vs wall-time metrics, tick counts, IPC,
        and any simulator-specific counters.  Displayed in the
        dashboard's session detail view.

        Parameters
        ----------
        session_id : str
            The session to query.

        Returns
        -------
        SessionPerf
            Performance snapshot including simulated time, wall time,
            ticks, IPC, and simulation speed ratio.

        Raises
        ------
        KeyError
            If no session with this ID exists.
        """
        ...

    # ------------------------------------------------------------------
    #  Run preparation
    # ------------------------------------------------------------------

    def get_exec_run_def(self, exec_def: ExecDef) -> RunDef:
        """Transform an execution request into a simulator-aware RunDef.

        Called by the daemon before starting every run inside a session.
        The simulator can:

        - Inject or override environment variables (e.g.
          ``ROCJITSU_CONFIG``, ``HSA_OVERRIDE_GFX_VERSION``).
        - Wrap or rewrite the command (e.g. prefix with a simulator
          launcher).
        - Append extra arguments.

        The returned :class:`RunDef` is what the daemon actually executes
        inside the container.

        Parameters
        ----------
        exec_def : ExecDef
            The original execution request containing a ``session`` id
            and the user's :class:`RunDef`.

        Returns
        -------
        RunDef
            A (possibly modified) run definition with simulator-specific
            env vars and command transformations applied.
        """
        ...

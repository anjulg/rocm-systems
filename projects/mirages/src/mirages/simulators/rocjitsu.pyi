"""rocjitsu simulator plugin for mirages.

rocjitsu is an AMD GPU simulator that decodes and executes actual GPU
machine code (CDNA3 gfx942 / CDNA4 gfx950, and experimental RISC-V)
within a discrete-event simulation framework (``simdojo``).

Interposition strategy
----------------------
rocjitsu intercepts GPU workloads via ``LD_PRELOAD`` of
``librocjitsu.so``.  When preloaded into a HIP / PyTorch / vLLM
process, it intercepts HSA runtime calls and redirects kernel
dispatches to the simulated GPU.  The simulator is configured
entirely through files and environment variables — no code changes
are needed in the target application.

Container layout
----------------
When :meth:`create_session` builds a :class:`ContainerDef`, the
following are injected:

.. list-table::
   :header-rows: 1

   * - Container path
     - Content
     - Purpose
   * - ``/opt/rocjitsu/lib/librocjitsu.so``
     - bind-mount from host
     - Interposition shared library
   * - ``/opt/rocjitsu/config/simulation.json``
     - injected file (generated)
     - Simulation config (topology, exec mode, threads)
   * - ``/opt/rocjitsu/config/simulation_config.fbs``
     - injected file
     - FlatBuffers schema for validation

Environment variables set in the container:

.. list-table::
   :header-rows: 1

   * - Variable
     - Value
     - Purpose
   * - ``LD_PRELOAD``
     - ``/opt/rocjitsu/lib/librocjitsu.so``
     - Intercept HSA runtime calls
   * - ``ROCJITSU_CONFIG``
     - ``/opt/rocjitsu/config/simulation.json``
     - Path to simulation config inside container
   * - ``ROCJITSU_SCHEMA``
     - ``/opt/rocjitsu/config/simulation_config.fbs``
     - FlatBuffers schema path for config validation
   * - ``ROCJITSU_EXEC_MODE``
     - ``functional`` | ``clocked``
     - Simulation execution mode
   * - ``HSA_OVERRIDE_GFX_VERSION``
     - e.g. ``9.4.2``
     - Tells the ROCm stack which GPU version is present

Supported GPUs
--------------
+------------+--------+----------------------------------------------+
| Name       | Arch   | Description                                  |
+============+========+==============================================+
| MI300X     | gfx942 | CDNA3 — 8 XCDs, 4 SEs/XCD, 8 CUs/SE        |
+------------+--------+----------------------------------------------+
| MI325X     | gfx950 | CDNA4 — 8 XCDs, 4 SEs/XCD, 8 CUs/SE        |
+------------+--------+----------------------------------------------+

Custom GPU design
-----------------
rocjitsu supports custom GPU topologies.  The ``topology_json`` field
of :class:`CustomGpuDef` accepts a JSON subtree matching the
``TopologyDef`` table in ``experimental/rocjitsu/schemas/simulation_config.fbs``:

.. code-block:: json

   {
     "root": {
       "name": "soc", "type": "soc",
       "children": [
         {"name": "vram", "type": "gpu_memory"},
         {
           "name": "xcd[0:N]", "type": "xcd",
           "children": [
             {"name": "l2", "type": "l2_cache"},
             {"name": "cp", "type": "command_processor"},
             {
               "name": "se[0:M]", "type": "shader_engine",
               "children": [{
                 "name": "cu[0:K]", "type": "compute_unit",
                 "config": [
                   {"key": "num_wf_slots", "value": "10"},
                   {"key": "sgprs_per_wf", "value": "104"},
                   {"key": "vgprs_per_wf", "value": "256"},
                   {"key": "lds_size_kb", "value": "160"}
                 ]
               }]
             }
           ]
         }
       ]
     },
     "links": [...]
   }

Where ``N`` = number of XCDs, ``M`` = SEs per XCD, ``K`` = CUs per SE.
"""

from __future__ import annotations

from mirages.simulator import Simulator
from mirages.types import (
    BindMount,
    ContainerDef,
    CustomGpuDef,
    ExecDef,
    GpuDef,
    GpuFamily,
    HealthStatus,
    InjectedFile,
    ProfileDef,
    RunDef,
    SessionDef,
    SessionHealth,
    SessionPerf,
    SetEnv,
    SimulatorInfo,
    SimulatorMode,
    Time,
)


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

#: Filesystem path inside the container where librocjitsu.so is mounted.
LIB_CONTAINER_PATH: str

#: Filesystem path inside the container for the generated simulation config.
CONFIG_CONTAINER_PATH: str

#: Filesystem path inside the container for the FlatBuffers schema file.
SCHEMA_CONTAINER_PATH: str

#: Host path to the compiled librocjitsu.so shared library.
#: Resolved at daemon startup from the ``rocjitsu`` package installation.
LIB_HOST_PATH: str

#: Default max simulation ticks (100 000).
DEFAULT_MAX_TICKS: int

#: Default number of simulation threads (1 = single-threaded).
DEFAULT_NUM_THREADS: int


class RocjitsuSimulator:
    """rocjitsu simulator plugin — implements :class:`~mirages.simulator.Simulator`.

    Stateful: tracks active sessions by session ID.

    Instantiated by the daemon via the ``mirages.simulators`` entry-point::

        [project.entry-points."mirages.simulators"]
        rocjitsu = "mirages.simulators.rocjitsu:RocjitsuSimulator"
    """

    def __init__(self) -> None:
        """Initialise the rocjitsu simulator.

        Locates ``librocjitsu.so`` on the host and reads the bundled
        FlatBuffers schema (``simulation_config.fbs``) for topology
        validation.
        """
        ...

    # ------------------------------------------------------------------
    #  Identity
    # ------------------------------------------------------------------

    def info(self) -> SimulatorInfo:
        """Return rocjitsu metadata.

        Returns
        -------
        SimulatorInfo
            - ``name``: ``"rocjitsu"``
            - ``version``: current rocjitsu library version
            - ``description``: one-liner about the simulator
            - ``supported_gpus``: ``[MI300X (gfx942), MI325X (gfx950)]``
            - ``supports_custom_gpus``: ``True``
            - ``supported_modes``: ``[Functional, Clocked]``
        """
        ...

    # ------------------------------------------------------------------
    #  GPU discovery & customisation
    # ------------------------------------------------------------------

    def supported_gpus(self) -> list[GpuDef]:
        """Return pre-defined GPU models.

        Returns
        -------
        list[GpuDef]
            Two entries:

            - ``GpuDef(name="MI300X", arch="gfx942", family=AmdCdna)``
            - ``GpuDef(name="MI325X", arch="gfx950", family=AmdCdna)``
        """
        ...

    def supports_custom_gpus(self) -> bool:
        """Always returns ``True``.

        rocjitsu allows fully custom GPU topologies — users can configure
        the number of XCDs, shader engines per XCD, compute units per SE,
        and per-CU parameters (wavefront slots, register counts, LDS size).
        """
        ...

    def set_custom_gpu(self, gpu: CustomGpuDef) -> GpuDef:
        """Validate and register a custom GPU topology.

        Parses ``gpu.topology_json`` against the ``TopologyDef`` table
        from ``simulation_config.fbs``.  If ``gpu.base_gpu`` is set,
        starts from that GPU's default topology and merges overrides.

        Validation checks:
        - Component ``type`` values are known (soc, xcd, shader_engine,
          compute_unit, l2_cache, command_processor, gpu_memory).
        - Range expressions (e.g. ``xcd[0:8]``) parse correctly.
        - Config keys are valid for their component type.
        - Link patterns reference existing components.

        Parameters
        ----------
        gpu : CustomGpuDef
            Custom GPU definition with ``topology_json`` describing
            the component hierarchy and links.

        Returns
        -------
        GpuDef
            A new GPU definition with ``family=AmdCdna`` and
            ``arch`` inferred from the topology (gfx942 or gfx950).

        Raises
        ------
        ValueError
            If ``topology_json`` is malformed or fails validation.
        """
        ...

    # ------------------------------------------------------------------
    #  Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self, session: SessionDef, profile: ProfileDef
    ) -> ContainerDef:
        """Create a rocjitsu simulation session.

        Generates a simulation config JSON tailored to the given profile:

        - ``exec_mode`` from ``profile.mode`` (Functional or Clocked)
        - ``vm.arch`` from the GPU's architecture string (gfx942, gfx950)
        - ``topology`` from the GPU's default or custom topology, scaled
          by ``profile.num_gpus`` (affects how many SoCs are instantiated)
        - ``num_threads`` defaults to 1 (single-event-queue)

        The returned :class:`ContainerDef` includes:

        - **env**: ``LD_PRELOAD``, ``ROCJITSU_CONFIG``,
          ``ROCJITSU_SCHEMA``, ``ROCJITSU_EXEC_MODE``,
          ``HSA_OVERRIDE_GFX_VERSION``
        - **mounts**: ``librocjitsu.so`` from host
        - **injected_files**: generated ``simulation.json`` +
          ``simulation_config.fbs`` schema

        Parameters
        ----------
        session : SessionDef
            The session being created.
        profile : ProfileDef
            Resolved profile with GPU, mode, and cluster shape.

        Returns
        -------
        ContainerDef
            Ready-to-launch container specification.
        """
        ...

    def delete_session(self, session_id: str) -> None:
        """Clean up a rocjitsu session.

        Removes internal bookkeeping state and any temp files generated
        for the session's simulation config.

        Parameters
        ----------
        session_id : str
            Session to tear down.

        Raises
        ------
        KeyError
            If the session does not exist.
        """
        ...

    # ------------------------------------------------------------------
    #  Session observability
    # ------------------------------------------------------------------

    def session_health(self, session_id: str) -> SessionHealth:
        """Check if the rocjitsu simulation engine is responsive.

        Probes the simulation process inside the container.  Returns
        ``Healthy`` when the simdojo event loop is running or idle
        (waiting for dispatches), and ``Unhealthy`` if the process
        has crashed or is unresponsive.

        The ``diagnostics_json`` field may include:

        - ``event_queue_depth``: number of pending events
        - ``active_wavefronts``: number of executing wavefronts
        - ``memory_usage_mb``: RSS of the simulation process

        Parameters
        ----------
        session_id : str
            Session to check.

        Returns
        -------
        SessionHealth
        """
        ...

    def session_perf(self, session_id: str) -> SessionPerf:
        """Query rocjitsu performance counters.

        Reads simulation statistics from the rocjitsu C API:

        - ``ticks``: simdojo discrete-event ticks elapsed
        - ``simulated_time``: simulated GPU clock time
        - ``wall_time``: real wall-clock elapsed since session start
        - ``ipc``: instructions per cycle across all active CUs
        - ``simulation_speed``: ratio of simulated to wall time
        - ``active_contexts``: number of active wavefronts

        The ``extra_json`` field may include:

        - ``dispatches_completed``: total kernel dispatches finished
        - ``cache_hit_rate_l2``: L2 cache hit ratio
        - ``memory_bandwidth_gb_s``: simulated HBM bandwidth

        Parameters
        ----------
        session_id : str
            Session to query.

        Returns
        -------
        SessionPerf
        """
        ...

    # ------------------------------------------------------------------
    #  Run preparation
    # ------------------------------------------------------------------

    def get_exec_run_def(self, exec_def: ExecDef) -> RunDef:
        """Prepare a run for execution inside a rocjitsu session.

        Takes the user's original :class:`ExecDef` and produces a
        :class:`RunDef` with rocjitsu-specific adjustments:

        1. Merges the session's simulator env vars (``LD_PRELOAD``,
           ``ROCJITSU_CONFIG``, etc.) into ``RunDef.env``.
        2. If the exec mode is ``Clocked``, may set additional env vars
           like ``ROCJITSU_MAX_TICKS`` for the specific run.
        3. The command and args are passed through unmodified — rocjitsu
           intercepts via ``LD_PRELOAD``, not by wrapping commands.

        Parameters
        ----------
        exec_def : ExecDef
            The original execution request.

        Returns
        -------
        RunDef
            The hydrated run definition ready for container execution.
        """
        ...

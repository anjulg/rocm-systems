"""Python type stubs mirroring the FlatBuffer tables in ``schema/``.

These are thin typed wrappers around the FlatBuffer-generated code.
Each class here corresponds 1-to-1 with a FlatBuffer ``table`` or ``enum``
defined in ``common.fbs`` or ``simulator.fbs``.  Field names and types
match the schema exactly.

These stubs exist so that:

1. Simulator authors get full IDE autocompletion and type checking
   without running ``flatc`` first.
2. The :class:`~mirages.simulator.Simulator` protocol can reference
   concrete types rather than raw ``bytes`` buffers.

At runtime the daemon may use the actual FlatBuffer-generated classes
or these dataclass-style wrappers — the two are kept in sync by CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


# ===================================================================
#  Enums  (from common.fbs + simulator.fbs)
# ===================================================================


class GpuFamily(IntEnum):
    """The family or vendor of a GPU implementation.

    Maps to ``mirages.GpuFamily`` in ``common.fbs``.
    """
    Unknown = 0
    AmdCdna = 1
    AmdRdna = 2
    RiscV = 3


class SimulatorMode(IntEnum):
    """Trade-off between simulation fidelity and performance.

    Maps to ``mirages.SimulatorMode`` in ``common.fbs``.
    """
    Functional = 0
    Clocked = 1
    CycleAccurate = 2


class Stream(IntEnum):
    """I/O stream identifiers used in :class:`StreamData`.

    Maps to ``mirages.Stream`` in ``common.fbs``.
    """
    Stdin = 0
    Stdout = 1
    Stderr = 2


class HealthStatus(IntEnum):
    """Health status of a session or component.

    Maps to ``mirages.HealthStatus`` in ``common.fbs``.
    """
    Unknown = 0
    Healthy = 1
    Unhealthy = 2


# ===================================================================
#  Structs  (from common.fbs)
# ===================================================================


@dataclass(frozen=True)
class Time:
    """Picosecond-accurate timestamp.

    Maps to ``mirages.Time`` struct in ``common.fbs``.
    """
    seconds: int = 0
    picoseconds: int = 0


# ===================================================================
#  Tables  (from common.fbs)
# ===================================================================


@dataclass
class SetEnv:
    """A single environment variable (key=value).

    Maps to ``mirages.SetEnv`` in ``common.fbs``.
    """
    key: str
    value: str


@dataclass
class GpuDef:
    """A GPU model a simulator can emulate.

    Maps to ``mirages.GpuDef`` in ``common.fbs``.
    """
    name: str
    arch: str
    family: GpuFamily = GpuFamily.Unknown
    description: str | None = None


@dataclass
class ProfileDef:
    """Named configuration binding simulator + GPU + mode + cluster shape.

    Maps to ``mirages.ProfileDef`` in ``common.fbs``.
    """
    name: str
    simulator: str
    gpu: str
    mode: SimulatorMode = SimulatorMode.Functional
    num_gpus: int = 1
    num_nodes: int = 1


@dataclass
class RunDef:
    """A command to execute, with optional args and env vars.

    Maps to ``mirages.RunDef`` in ``common.fbs``.
    """
    command: str
    args: list[str] = field(default_factory=list)
    env: list[SetEnv] = field(default_factory=list)


@dataclass
class RunExit:
    """How a run ended.

    Maps to ``mirages.RunExit`` in ``common.fbs``.
    """
    run_id: str | None = None
    exit_code: int = 0


@dataclass
class StreamData:
    """A chunk of I/O data from a stream.

    Maps to ``mirages.StreamData`` in ``common.fbs``.
    """
    stream: Stream = Stream.Stdout
    data: bytes = b""


@dataclass
class ExecDef:
    """Request to execute a command inside a session.

    Maps to ``mirages.ExecDef`` in ``common.fbs``.
    """
    session: str
    run: RunDef


@dataclass
class SessionDef:
    """A session — a running instance of a profile in a container.

    Maps to ``mirages.SessionDef`` in ``common.fbs``.
    """
    name: str
    profile: str
    image: str | None = None


@dataclass
class ClusterDef:
    """A cluster of sessions connected for multi-node simulation.

    Maps to ``mirages.ClusterDef`` in ``common.fbs``.
    """
    head_address: str
    sessions: list[str] = field(default_factory=list)


# ===================================================================
#  Tables  (from simulator.fbs)
# ===================================================================


@dataclass
class SimulatorInfo:
    """Metadata a simulator returns to identify itself.

    Maps to ``mirages.simulator.SimulatorInfo`` in ``simulator.fbs``.
    """
    name: str
    version: str
    supported_gpus: list[GpuDef]
    description: str | None = None
    supports_custom_gpus: bool = False
    supported_modes: list[SimulatorMode] = field(default_factory=list)


@dataclass
class CustomGpuDef:
    """A user-designed custom GPU topology.

    Maps to ``mirages.simulator.CustomGpuDef`` in ``simulator.fbs``.
    """
    name: str
    topology_json: str
    base_gpu: str | None = None


@dataclass
class BindMount:
    """A host-to-container bind mount.

    Maps to ``mirages.simulator.BindMount`` in ``simulator.fbs``.
    """
    host_path: str
    container_path: str
    readonly: bool = True


@dataclass
class InjectedFile:
    """A file generated at session-creation time, injected into the container.

    Maps to ``mirages.simulator.InjectedFile`` in ``simulator.fbs``.
    """
    container_path: str
    content: bytes
    readonly: bool = True
    label: str | None = None


@dataclass
class PortMapping:
    """A port mapping from container to host.

    Maps to ``mirages.simulator.PortMapping`` in ``simulator.fbs``.
    """
    container_port: int
    host_port: int = 0
    protocol: str | None = None
    label: str | None = None


@dataclass
class ContainerDef:
    """Full OCI container specification for a simulation session.

    Maps to ``mirages.simulator.ContainerDef`` in ``simulator.fbs``.
    """
    image: str | None = None
    env: list[SetEnv] = field(default_factory=list)
    mounts: list[BindMount] = field(default_factory=list)
    injected_files: list[InjectedFile] = field(default_factory=list)
    entrypoint: list[str] = field(default_factory=list)
    working_dir: str | None = None
    ports: list[PortMapping] = field(default_factory=list)
    privileged: bool = False
    resource_limits_json: str | None = None


@dataclass
class SessionHealth:
    """Health report for a running session.

    Maps to ``mirages.simulator.SessionHealth`` in ``simulator.fbs``.
    """
    session_id: str
    status: HealthStatus = HealthStatus.Unknown
    uptime: Time | None = None
    error_message: str | None = None
    diagnostics_json: str | None = None


@dataclass
class SessionPerf:
    """Performance counters for a running session.

    Maps to ``mirages.simulator.SessionPerf`` in ``simulator.fbs``.
    """
    session_id: str
    simulated_time: Time | None = None
    wall_time: Time | None = None
    ticks: int = 0
    ipc: float = 0.0
    simulation_speed: float = 0.0
    active_contexts: int = 0
    extra_json: str | None = None

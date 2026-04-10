"""Simulator plugin registry.

The daemon discovers simulators via Python entry-points registered under
the ``mirages.simulators`` group.  Third-party packages can register
their own simulators by adding an entry-point in their ``pyproject.toml``::

    [project.entry-points."mirages.simulators"]
    my_sim = "my_package.sim:MySimulator"

At daemon startup, :func:`load_simulators` iterates over all registered
entry-points, instantiates each one, and verifies it satisfies the
:class:`~mirages.simulator.Simulator` protocol.
"""

from mirages.simulator import Simulator


def load_simulators() -> dict[str, Simulator]:
    """Discover and instantiate all registered simulator plugins.

    Scans the ``mirages.simulators`` entry-point group, imports each
    entry-point, instantiates it (calling with no arguments), and
    validates that the resulting object satisfies the
    :class:`~mirages.simulator.Simulator` protocol.

    Returns
    -------
    dict[str, Simulator]
        A mapping from ``SimulatorInfo.name`` to simulator instance.

    Raises
    ------
    TypeError
        If an entry-point does not satisfy the Simulator protocol.
    """
    ...


def get_simulator(name: str) -> Simulator:
    """Look up a loaded simulator by its registered name.

    Parameters
    ----------
    name : str
        The ``SimulatorInfo.name`` of the simulator (e.g. ``"rocjitsu"``).

    Returns
    -------
    Simulator

    Raises
    ------
    KeyError
        If no simulator with this name is registered.
    """
    ...

from typing import TYPE_CHECKING, Protocol

from ares.managers.manager_mediator import ManagerMediator
from sc2.unit import Unit
from sc2.units import Units

if TYPE_CHECKING:
    from ares import AresBot


class BaseCombat(Protocol):
    """Basic interface that all combat classes should follow.

    Parameters
    ----------
    ai : AresBot
        Bot object that will be running the game
    config : Dict[Any, Any]
        Dictionary with the data from the configuration file
    mediator : ManagerMediator         u
        Used for getting information from managers in Ares.
    """

    ai: "AresBot"
    config: dict
    mediator: ManagerMediator

    def execute(self, units: Units | list[Unit], **kwargs) -> None:
        """Execute the implemented behavior.

        This should be called every step.

        Parameters
        ----------
        units : Units | list[Unit]
            The exact units that will be controlled by
            the implemented `BaseUnit` class.
        **kwargs :
            See combat subclasses docstrings for supported kwargs.

        """
        ...

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import KeepUnitSafe, PathUnitToTarget, UseAbility
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions.geometry import cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat

if TYPE_CHECKING:
    from ares import AresBot

DANGER_TO_AIR: set[UnitID] = {
    UnitID.VOIDRAY,
    UnitID.PHOTONCANNON,
    UnitID.MISSILETURRET,
    UnitID.SPORECRAWLER,
    UnitID.BUNKER,
}


@dataclass
class SCVProxyBuilder(BaseCombat):
    """Execute behavior for scv building proxies.

    Parameters
    ----------
    ai : AresBot
        Bot object that will be running the game
    config : Dict[Any, Any]
        Dictionary with the data from the configuration file
    mediator : ManagerMediator
        Used for getting information from managers in Ares.
    """

    ai: "AresBot"
    config: dict
    mediator: ManagerMediator

    def execute(self, units: Union[list[Unit], Units], **kwargs) -> None:
        target: Point2 = kwargs["target"]
        next_item_to_build: UnitTypeId | None = kwargs["next_item_to_build"]
        build_location: Point2 | None = kwargs["build_location"]
        avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid
        grid: np.ndarray = self.mediator.get_ground_grid
        ability_id: AbilityId | None = None
        if next_item_to_build:
            ability_id = self.ai.game_data.units[
                next_item_to_build.value
            ].creation_ability.id

        for unit in units:
            if any(o.ability.id == ability_id for o in unit.orders):
                continue

            proxy_maneuver: CombatManeuver = CombatManeuver()
            proxy_maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
            # attacking_maneuver.add(ShootTargetInRange(unit, close_enemy))

            if next_item_to_build and build_location:
                if cy_distance_to_squared(unit.position, build_location) < 10:
                    ability: AbilityId = self.ai.game_data.units[
                        next_item_to_build.value
                    ].creation_ability.id
                    proxy_maneuver.add(
                        UseAbility(ability, unit=unit, target=build_location)
                    )
                else:
                    proxy_maneuver.add(
                        PathUnitToTarget(unit=unit, target=build_location, grid=grid)
                    )
            else:
                if unit.tag != primary_builder_tag or (
                    not build_location
                    and cy_distance_to_squared(unit.position, target) < 100.0
                ):
                    proxy_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                proxy_maneuver.add(
                    PathUnitToTarget(
                        unit=unit, target=target, grid=grid, success_at_distance=4.0
                    )
                )
                if unit.tag == primary_builder_tag:
                    proxy_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
            self.ai.register_behavior(proxy_maneuver)

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import KeepUnitSafe, PathUnitToTarget, UseAbility
from ares.consts import UnitTreeQueryType
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_distance_to_squared
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES

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
class BattleCruiserCombat(BaseCombat):
    """Execute behavior for Tempest Combat.

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
        target: Point2 = (
            kwargs["target"] if "target" in kwargs else self.ai.enemy_start_locations[0]
        )
        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=units,
            distances=13,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        avoid_grid: np.ndarray = self.mediator.get_air_avoidance_grid
        grid: np.ndarray = self.mediator.get_air_grid

        for unit in units:
            dist_to_target: float = cy_distance_to_squared(unit.position, target)
            jump_ready: bool = AbilityId.EFFECT_TACTICALJUMP in unit.abilities
            close_enemy: list[Unit] = [
                u
                for u in near_enemy[unit.tag]
                if (not u.is_cloaked or u.is_cloaked and u.is_revealed)
                and (not u.is_burrowed or u.is_burrowed and u.is_visible)
                and not u.is_memory
                and not u.is_snapshot
                and u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]

            attacking_maneuver: CombatManeuver = CombatManeuver()
            attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))
            if jump_ready and dist_to_target > 2500:
                jump_spot: Point2 = self.mediator.find_closest_safe_spot(
                    from_pos=target, grid=grid, radius=10.0
                )
                attacking_maneuver.add(
                    UseAbility(
                        unit=unit,
                        ability=AbilityId.EFFECT_TACTICALJUMP,
                        target=jump_spot,
                    )
                )
            elif unit.health < 225.0:
                attacking_maneuver.add(
                    PathUnitToTarget(
                        unit=unit, target=self.ai.main_base_ramp.top_center, grid=grid
                    )
                )
            else:
                if close_enemy:
                    attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))

                attacking_maneuver.add(
                    PathUnitToTarget(unit=unit, target=target, grid=grid)
                )
            self.ai.register_behavior(attacking_maneuver)

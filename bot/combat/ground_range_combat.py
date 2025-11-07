from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from sc2.ids.ability_id import AbilityId

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    StutterUnitBack,
    StutterUnitForward,
    AMove,
)
from ares.consts import ALL_STRUCTURES, UnitTreeQueryType
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to, cy_is_facing
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

STRUCTURE_TARGETS: set[UnitID] = {
    UnitID.BUNKER,
    UnitID.PLANETARYFORTRESS,
    UnitID.SPINECRAWLER,
    UnitID.PHOTONCANNON,
    UnitID.PYLON,
    UnitID.SUPPLYDEPOT,
    UnitID.SUPPLYDEPOTLOWERED,
    UnitID.OVERLORD,
    UnitID.OVERLORDTRANSPORT,
    UnitID.OVERSEER,
    UnitID.OVERSEERSIEGEMODE,
    UnitID.OVERLORDCOCOON,
}


@dataclass
class GroundRangeCombat(BaseCombat):
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
        everything_near_squad: Units = kwargs["everything_near_squad"]
        target: Point2 = (
            kwargs["target"] if "target" in kwargs else self.ai.enemy_start_locations[0]
        )
        close_enemy: list[Unit] = [
            u
            for u in everything_near_squad
            if (not u.is_cloaked or u.is_cloaked and u.is_revealed)
            and (not u.is_burrowed or u.is_burrowed and u.is_visible)
            and not u.is_memory
            and not u.is_snapshot
            and u.type_id not in COMMON_UNIT_IGNORE_TYPES
        ]
        priority_units: list[Unit] = [
            u
            for u in close_enemy
            if u.type_id not in ALL_STRUCTURES or u.type_id in STRUCTURE_TARGETS
        ]
        avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid
        grid: np.ndarray = self.mediator.get_ground_grid
        can_engage: Point2 = kwargs["can_engage"]
        squad_position: Point2 = kwargs["squad_position"]

        for unit in units:
            if unit.type_id == UnitID.MEDIVAC:
                self._handle_medivac(unit, target, squad_position, units)
                continue
            if not unit.can_attack_air:
                priority_units: list[Unit] = [
                    u for u in priority_units if not u.is_flying
                ]
            attacking_maneuver: CombatManeuver = CombatManeuver()
            attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=avoid_grid))

            attacking_maneuver.add(ShootTargetInRange(unit, priority_units))
            if not priority_units:
                attacking_maneuver.add(ShootTargetInRange(unit, close_enemy))

            if can_engage:
                if close_enemy:
                    target_unit: Unit | None
                    if priority_units:
                        target_unit = cy_closest_to(unit.position, priority_units)
                    else:
                        target_unit = cy_closest_to(unit.position, close_enemy)

                    if (
                        not target_unit.is_flying
                        and target_unit.ground_range > unit.ground_range
                    ):
                        attacking_maneuver.add(
                            StutterUnitForward(unit=unit, target=target_unit)
                        )
                    else:
                        attacking_maneuver.add(
                            StutterUnitBack(unit=unit, target=target_unit, grid=grid)
                        )

                else:
                    attacking_maneuver.add(
                        PathUnitToTarget(
                            unit=unit, target=target, grid=grid, success_at_distance=5.0
                        )
                    )
            else:
                attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
                attacking_maneuver.add(
                    PathUnitToTarget(
                        unit=unit, target=target, grid=grid, success_at_distance=5.0
                    )
                )
            self.ai.register_behavior(attacking_maneuver)

    def _handle_medivac(self, unit, target, squad_position, units):
        if unit.is_using_ability(AbilityId.MEDIVACHEAL_HEAL):
            return
        grid: np.ndarray = self.ai.mediator.get_air_grid
        other_units: list[Unit] = [
            u for u in units if u.type_id in {UnitID.MARINE, UnitID.MARAUDER}
        ]
        if len(other_units) > 0:
            _target: Point2 = squad_position
        else:
            _target: Point2 = target

        attacking_maneuver: CombatManeuver = CombatManeuver()
        if len(other_units) == 0:
            attacking_maneuver.add(KeepUnitSafe(unit=unit, grid=grid))
            attacking_maneuver.add(
                PathUnitToTarget(unit=unit, grid=grid, target=_target)
            )
        else:
            attacking_maneuver.add(AMove(unit, _target))
        self.ai.register_behavior(attacking_maneuver)

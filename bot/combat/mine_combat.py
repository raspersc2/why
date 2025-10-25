from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
from sc2.ids.ability_id import AbilityId
from sc2.ids.upgrade_id import UpgradeId

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe,
    PathUnitToTarget,
    ShootTargetInRange,
    StutterUnitBack,
    StutterUnitForward,
    UseAbility,
)
from ares.consts import ALL_STRUCTURES, UnitTreeQueryType
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import cy_closest_to, cy_distance_to_squared
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
class MineCombat(BaseCombat):
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
        stay_burrowed: bool = (
            kwargs["stay_burrowed"] if "stay_burrowed" in kwargs else False
        )
        near_enemy: dict[int, Units] = self.mediator.get_units_in_range(
            start_points=units,
            distances=13,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=True,
        )
        avoid_grid: np.ndarray = self.mediator.get_ground_avoidance_grid
        grid: np.ndarray = self.mediator.get_ground_grid

        drilling_claws_available: bool = UpgradeId.DRILLCLAWS in self.ai.state.upgrades
        ability: AbilityId = AbilityId.WIDOWMINEATTACK_WIDOWMINEATTACK
        current_frame: int = self.ai.state.game_loop
        unit_to_ability_dict: dict[
            int, dict[AbilityId, int]
        ] = self.ai.mediator.get_unit_to_ability_dict

        for unit in units:
            attack_available: bool = (
                current_frame >= unit_to_ability_dict[unit.tag][ability]
            )
            close_enemy: list[Unit] = [
                u
                for u in near_enemy[unit.tag]
                if u.type_id not in COMMON_UNIT_IGNORE_TYPES
            ]

            only_enemy_units: list[Unit] = [
                u for u in close_enemy if u.type_id not in ALL_STRUCTURES
            ]

            attacking_maneuver: CombatManeuver = CombatManeuver()
            if unit.is_burrowed:
                attacking_maneuver.add(
                    self._burrowed_mine_behavior(
                        unit,
                        only_enemy_units,
                        grid,
                        avoid_grid,
                        attack_available,
                        drilling_claws_available,
                        target,
                        stay_burrowed,
                    )
                )
            else:
                attacking_maneuver.add(
                    self._unburrowed_mine_behavior(
                        unit,
                        only_enemy_units,
                        grid,
                        avoid_grid,
                        attack_available,
                        drilling_claws_available,
                        target,
                    )
                )
            self.ai.register_behavior(attacking_maneuver)

    def _burrowed_mine_behavior(
        self,
        unit: Unit,
        only_enemy_units: list[Unit],
        grid: np.ndarray,
        avoid_grid: np.ndarray,
        attack_available: bool,
        drilling_claws_available: bool,
        target: Point2,
        stay_burrowed: bool,
    ) -> CombatManeuver:
        burrowed_mine_maneuver: CombatManeuver = CombatManeuver()
        if only_enemy_units:
            if attack_available:
                return burrowed_mine_maneuver
            elif drilling_claws_available and self.ai.mediator.get_is_detected(
                unit=unit
            ):
                burrowed_mine_maneuver.add(
                    UseAbility(AbilityId.BURROWUP_WIDOWMINE, unit)
                )
        else:
            if (
                not stay_burrowed
                and cy_distance_to_squared(unit.position, target) > 100.0
            ):
                burrowed_mine_maneuver.add(
                    UseAbility(AbilityId.BURROWUP_WIDOWMINE, unit)
                )
        return burrowed_mine_maneuver

    def _unburrowed_mine_behavior(
        self,
        unit: Unit,
        only_enemy_units: list[Unit],
        grid: np.ndarray,
        avoid_grid: np.ndarray,
        attack_available: bool,
        drilling_claws_available: bool,
        target: Point2,
    ) -> CombatManeuver:
        aggressive: bool = drilling_claws_available
        unburrowed_mine_maneuver: CombatManeuver = CombatManeuver()
        if only_enemy_units:
            if aggressive and attack_available:
                in_range: bool = (
                    len(
                        [
                            u
                            for u in only_enemy_units
                            if cy_distance_to_squared(unit.position, u.position) < 25.56
                        ]
                    )
                    > 0
                )
                if in_range:
                    unburrowed_mine_maneuver.add(
                        UseAbility(AbilityId.BURROWDOWN_WIDOWMINE, unit)
                    )
                else:
                    enemy_target: Point2 = cy_closest_to(
                        unit.position, only_enemy_units
                    )
                    unburrowed_mine_maneuver.add(
                        UseAbility(AbilityId.MOVE_MOVE, unit, enemy_target)
                    )
            else:
                if attack_available or (
                    drilling_claws_available
                    and not self.ai.mediator.get_is_detected(unit=unit)
                ):
                    unburrowed_mine_maneuver.add(
                        UseAbility(AbilityId.BURROWDOWN_WIDOWMINE, unit)
                    )
                else:
                    unburrowed_mine_maneuver.add(KeepUnitSafe(unit, grid=grid))
                    unburrowed_mine_maneuver.add(
                        UseAbility(AbilityId.BURROWDOWN_WIDOWMINE, unit)
                    )

        else:
            unburrowed_mine_maneuver.add(
                PathUnitToTarget(unit, grid, target, success_at_distance=5.0)
            )
            unburrowed_mine_maneuver.add(
                UseAbility(AbilityId.BURROWDOWN_WIDOWMINE, unit)
            )

        return unburrowed_mine_maneuver

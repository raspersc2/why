"""Behavior for harass Reaper."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AttackTarget,
    KeepUnitSafe,
    MoveToSafeTarget,
    PathUnitToTarget,
    ReaperGrenade,
    ShootTargetInRange,
    UseAbility,
    AMove,
)
from ares.consts import (
    ALL_STRUCTURES,
    CREEP_TUMOR_TYPES,
    LOSS_MARGINAL_OR_WORSE,
    WORKER_TYPES,
    EngagementResult,
)
from ares.managers.manager_mediator import ManagerMediator
from cython_extensions import (
    cy_center,
    cy_closest_to,
    cy_distance_to,
    cy_distance_to_squared,
)
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.consts import COMMON_UNIT_IGNORE_TYPES

if TYPE_CHECKING:
    from ares import AresBot

STATIC_DEFENCE: set[UnitID] = {
    UnitID.BUNKER,
    UnitID.PLANETARYFORTRESS,
    UnitID.SPINECRAWLER,
    UnitID.PHOTONCANNON,
}


@dataclass
class ReaperHarass(BaseCombat):
    """Execute behavior for Reaper harass.

    Called from `ReaperHarassManager`

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
    reaper_grenade_range: float = 5.0

    def execute(self, units: Units | list[Unit], **kwargs) -> None:
        """Execute the Reaper harass.

        Parameters
        ----------
        units : list[Unit]
            The units we want ReaperHarass to control.
        **kwargs :
            See below.

        Keyword Arguments
        -----------------
        harass_target : Point2
        heal_threshold: float
            Health percentage where a Reaper should disengage to heal
        force_fight: bool
            Reapers will take on any fight

        Returns
        -------

        """
        if not units:
            return

        harass_target: Point2 = kwargs["harass_target"]
        near_enemy: Units = kwargs["everything_near_reapers"]
        avoidance_grid = self.mediator.get_ground_avoidance_grid
        reaper_grid = self.mediator.get_climber_grid
        squad_pos: tuple = cy_center(units)

        only_threats: list[Unit] = [
            u
            for u in near_enemy
            if (
                (
                    u.type_id not in ALL_STRUCTURES
                    and (
                        u.type_id not in COMMON_UNIT_IGNORE_TYPES
                        or u.type_id == UnitID.MULE
                    )
                )
                or (u.type_id in STATIC_DEFENCE)
                or (u.type_id in CREEP_TUMOR_TYPES and u.is_active)
            )
        ]
        only_threats_without_memory: list[Unit] = [
            u for u in only_threats if not u.is_memory
        ]
        near_lings: list[Unit] = [
            t for t in only_threats_without_memory if t.type_id == UnitID.ZERGLING
        ]
        near_melee: list[Unit] = [
            u
            for u in only_threats_without_memory
            if u.ground_range < 3.0
            and cy_distance_to_squared(u.position, squad_pos) < 16.0
        ]
        near_workers: list[Unit] = [u for u in near_melee if u.type_id in WORKER_TYPES]
        # early game and single lings, ignore them
        if self.ai.time < 150.0 and len(near_lings) == 1 and len(only_threats) == 1:
            only_threats = []

        only_unit_threats_not_workers: list[Unit] = [
            u for u in only_threats if u.type_id not in WORKER_TYPES
        ]

        only_queens: bool = all(u.type_id == UnitID.QUEEN for u in only_threats)
        len_marines: int = len([u for u in only_threats if u.type_id == UnitID.MARINE])
        take_marine_fight: bool = 0 < len_marines <= len(units)

        heal_threshold: float = kwargs["heal_threshold"]
        if take_marine_fight:
            heal_threshold = 0.11

        if only_queens or take_marine_fight:
            fight_result: EngagementResult = EngagementResult.VICTORY_CLOSE
        else:
            fight_result: EngagementResult = self.mediator.can_win_fight(
                own_units=[
                    u
                    for u in self.ai.units
                    if cy_distance_to_squared(u.position, squad_pos) < 100
                    and u.type_id not in WORKER_TYPES
                ],
                enemy_units=only_unit_threats_not_workers,
            )

        _can_engage: bool = fight_result not in LOSS_MARGINAL_OR_WORSE

        for unit in units:
            target: Point2 = harass_target
            low_health: bool = unit.health_percentage <= heal_threshold
            grenade_targets: list[Unit] = [
                u
                for u in only_unit_threats_not_workers
                if cy_distance_to(u.position, unit.position)
                < self.reaper_grenade_range + unit.radius
            ]

            harass_maneuver: CombatManeuver = CombatManeuver()
            # dodge biles, storms etc
            harass_maneuver.add(KeepUnitSafe(unit=unit, grid=avoidance_grid))

            if not unit.is_attacking and [
                u
                for u in near_melee
                if cy_distance_to_squared(u.position, unit.position) < 6.5
            ]:
                harass_maneuver.add(KeepUnitSafe(unit=unit, grid=reaper_grid))
            # reaper grenade
            harass_maneuver.add(
                ReaperGrenade(
                    unit=unit,
                    enemy_units=grenade_targets,
                    retreat_target=self.mediator.get_own_nat,
                    grid=reaper_grid,
                    place_predictive=low_health,
                )
            )

            can_shoot: bool = True
            if low_health and not self.mediator.is_position_safe(
                grid=reaper_grid, position=unit.position
            ):
                can_shoot = False
            # don't hesitate shooting things
            if can_shoot:
                if near_workers:
                    harass_maneuver.add(
                        ShootTargetInRange(unit=unit, targets=near_workers)
                    )
                if only_threats:
                    harass_maneuver.add(
                        ShootTargetInRange(unit=unit, targets=only_threats)
                    )
                if not only_threats and near_enemy:
                    harass_maneuver.add(
                        ShootTargetInRange(unit=unit, targets=near_enemy)
                    )

            # low health and dangerous enemy, retreat and heal
            if low_health:
                harass_maneuver.add(
                    MoveToSafeTarget(unit, reaper_grid, self.ai.start_location)
                )

            # no enemies in sight, so let's sneak around
            elif not only_threats_without_memory:
                harass_maneuver.add(
                    PathUnitToTarget(
                        unit=unit,
                        grid=reaper_grid,
                        target=target,
                        success_at_distance=3.0,
                    )
                )

            else:
                # nearby tanks, try to get on top of them
                if len(only_unit_threats_not_workers) <= 3 and (
                    tanks := [
                        u
                        for u in only_unit_threats_not_workers
                        if u.type_id == UnitID.SIEGETANKSIEGED
                    ]
                ):
                    tank: Unit = cy_closest_to(unit.position, tanks)
                    harass_maneuver.add(
                        UseAbility(
                            ability=AbilityId.MOVE_MOVE,
                            unit=unit,
                            target=tank.position,
                        )
                    )

                # else micro vs any other unit type
                else:
                    if near_melee:
                        harass_maneuver.add(KeepUnitSafe(unit=unit, grid=reaper_grid))
                        harass_maneuver.add(
                            PathUnitToTarget(
                                unit=unit,
                                grid=reaper_grid,
                                target=cy_closest_to(
                                    unit.position, near_melee
                                ).position,
                            )
                        )
                    else:
                        if not _can_engage:
                            if only_threats_without_memory:
                                harass_maneuver.add(
                                    KeepUnitSafe(unit=unit, grid=reaper_grid)
                                )
                            harass_maneuver.add(
                                PathUnitToTarget(
                                    unit=unit,
                                    grid=reaper_grid,
                                    target=target,
                                    success_at_distance=5.0,
                                )
                            )

                        elif only_threats_without_memory:
                            harass_maneuver.add(
                                AttackTarget(
                                    unit=unit,
                                    target=cy_closest_to(
                                        unit.position, only_threats_without_memory
                                    ),
                                )
                            )
                        else:
                            harass_maneuver.add(AMove(unit=unit, target=target))

            harass_maneuver.add(KeepUnitSafe(unit=unit, grid=reaper_grid))

            self.ai.register_behavior(harass_maneuver)

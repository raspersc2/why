from ares import AresBot
from ares.consts import (
    ALL_STRUCTURES,
    LOSS_MARGINAL_OR_WORSE,
    VICTORY_CLOSE_OR_BETTER,
    EngagementResult,
    UnitRole,
    UnitTreeQueryType,
)
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_distance_to
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.base_combat import BaseCombat
from bot.combat.ground_range_combat import GroundRangeCombat
from bot.consts import BIO_FORCES, COMMON_UNIT_IGNORE_TYPES
from bot.openings.opening_base import OpeningBase

STATIC_DEFENCE: set[UnitTypeId] = {
    UnitTypeId.BUNKER,
    UnitTypeId.PLANETARYFORTRESS,
    UnitTypeId.SPINECRAWLER,
    UnitTypeId.PHOTONCANNON,
}


class Bio(OpeningBase):
    _ground_range_combat: BaseCombat

    SQUAD_ENGAGE_THRESHOLD: set[EngagementResult] = VICTORY_CLOSE_OR_BETTER
    SQUAD_DISENGAGE_THRESHOLD: set[EngagementResult] = LOSS_MARGINAL_OR_WORSE

    def __init__(self):
        super().__init__()

        self._squad_id_to_engage_tracker: dict = dict()

    async def on_start(self, ai: AresBot) -> None:
        await super().on_start(ai)
        self._ground_range_combat = GroundRangeCombat(ai, ai.config, ai.mediator)

    async def on_step(self, target: Point2 | None = None) -> None:
        self._micro(target)

    def _micro(self, attack_target: Point2 = None) -> None:
        if attack_target:
            squad_target = attack_target
        else:
            squad_target: Point2 = self.attack_target

        squads: list[UnitSquad] = self.ai.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=7.5
        )
        attackers: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.ATTACKING, unit_type=BIO_FORCES
        )
        if len(squads) > 0:
            pos_of_main_squad: Point2 = self.ai.mediator.get_position_of_main_squad(
                role=UnitRole.ATTACKING
            )

            for squad in squads:
                target: Point2
                if not squad.main_squad:
                    target = pos_of_main_squad
                else:
                    target = squad_target
                everything_near_squad: Units = (
                    self.ai.mediator.get_units_in_range(
                        start_points=[squad.squad_position],
                        distances=12.0,
                        query_tree=UnitTreeQueryType.AllEnemy,
                        return_as_dict=False,
                    )[0]
                ).filter(
                    lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES
                    or u.type_id == UnitTypeId.MULE
                )
                self._track_squad_engagement(attackers, squad)
                can_engage: bool = self._squad_id_to_engage_tracker[squad.squad_id]
                self._ground_range_combat.execute(
                    squad.squad_units,
                    everything_near_squad=everything_near_squad,
                    target=target,
                    can_engage=can_engage,
                    squad_position=squad.squad_position,
                )

    def _track_squad_engagement(self, attackers: Units, squad: UnitSquad) -> None:
        close_enemy: Units = self.ai.mediator.get_units_in_range(
            start_points=[squad.squad_position],
            distances=25.5,
            query_tree=UnitTreeQueryType.AllEnemy,
        )[0]
        only_units: list[Unit] = [
            u
            for u in close_enemy
            if u.type_id not in ALL_STRUCTURES or u.type_id in STATIC_DEFENCE
        ]
        squad_id: str = squad.squad_id
        if squad_id not in self._squad_id_to_engage_tracker:
            self._squad_id_to_engage_tracker[squad_id] = False

        # no enemy nearby, makes no sense to engage
        if not close_enemy:
            self._squad_id_to_engage_tracker[squad_id] = False
            return

        enemy_pos: Point2 = close_enemy.center
        tanks: list[Unit] = [
            u for u in squad.squad_units if u.type_id == UnitTypeId.SIEGETANKSIEGED
        ]

        # something near a sieged tank, engage
        for t in tanks:
            close_threats: list[Unit] = [
                e for e in only_units if cy_distance_to(e.position, t.position) < 8.5
            ]
            if len(close_threats) > 0:
                self._squad_id_to_engage_tracker[squad_id] = True
                return

        own_attackers_nearby: Units = attackers.filter(
            lambda a: cy_distance_to(a.position, enemy_pos) < 15.5
            and a.type_id not in {UnitTypeId.SIEGETANKSIEGED, UnitTypeId.SIEGETANK}
            and not a.is_flying
        )

        fight_result: EngagementResult = self.ai.mediator.can_win_fight(
            own_units=own_attackers_nearby, enemy_units=only_units
        )

        # currently engaging, see if we should disengage
        if self._squad_id_to_engage_tracker[squad.squad_id]:
            if fight_result in self.SQUAD_DISENGAGE_THRESHOLD:
                self._squad_id_to_engage_tracker[squad.squad_id] = False
        # not engaging, check if we can
        elif fight_result in self.SQUAD_ENGAGE_THRESHOLD:
            self._squad_id_to_engage_tracker[squad.squad_id] = True

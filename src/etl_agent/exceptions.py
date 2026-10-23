"""Классификация ошибок для политики повторов LangGraph."""


class PlanningError(Exception):
    """Постоянная ошибка провайдера или некорректный ответ."""


class TransientPlanningError(PlanningError):
    """Временная ошибка: сеть, ограничение частоты или 5xx."""


class PlanningTimeout(PlanningError):
    """Истекло время попытки вызова модели."""

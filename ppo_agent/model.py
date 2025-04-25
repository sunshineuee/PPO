import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PPOModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        """
        PPO модель с общей базовой сетью, разделяющейся на политику (actor) и ценностную функцию (critic).
        :param input_dim: количество входных признаков (фичей)
        :param hidden_dim: размер скрытого слоя
        :param output_dim: размер действия (например, сила сигнала: рост/падение)
        """
        super(PPOModel, self).__init__()

        # Общая часть
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (политика)
        self.policy_head = nn.Linear(hidden_dim, output_dim)  # Сигнал: [-1, 1]
        self.confidence_head = nn.Linear(hidden_dim, 1)       # Уверенность: [0, 1]

        # Critic (оценка состояния)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямой проход: возвращает предсказания для политики (сила сигнала), уверенности и ценности состояния.
        :param x: входной тензор признаков формы (batch_size, input_dim)
        :return: (signal, confidence, value)
        """
        base = self.shared(x)

        signal = torch.tanh(self.policy_head(base))              # значение сигнала: рост или падение [-1..1]
        confidence = torch.sigmoid(self.confidence_head(base))   # уверенность: [0..1]
        value = self.value_head(base)                             # ценность состояния

        return signal, confidence, value

    def get_policy(self, x: torch.Tensor) -> torch.Tensor:
        base = self.shared(x)
        return torch.tanh(self.policy_head(base))

    def get_confidence(self, x: torch.Tensor) -> torch.Tensor:
        base = self.shared(x)
        return torch.sigmoid(self.confidence_head(base))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        base = self.shared(x)
        return self.value_head(base)

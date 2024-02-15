#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        conv0 = nn.Conv2d(3, 128, 3, stride=1, padding=0)
        act0 = nn.LeakyReLU(0.2)
        maxpool0 = nn.MaxPool2d(2, 2)

        conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        act1 = nn.LeakyReLU(0.2)
        maxpool1 = nn.MaxPool2d(2, 2)

        conv2 = nn.Conv2d(128, 128, 3, stride=1, padding=0)
        act2 = nn.LeakyReLU(0.2)

        conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=0)
        act3 = nn.LeakyReLU(0.2)
        adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        flatten = nn.Flatten()
        linear1 = nn.Linear(256, 20)
        linear2 = nn.Linear(20, 2)

        self.model = nn.Sequential(
            conv0, act0, maxpool0,
            conv1, act1, maxpool1,
            conv2, act2,
            conv3, act3, adaptivepool,
            flatten, linear1, act0, linear2
        )

    def forward(self, x):
        x = self.model(x)

        return x





def main():
    """Run administrative tasks."""

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()

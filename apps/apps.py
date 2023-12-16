from django.apps import AppConfig
from modules.services.deriv_service import DerivService
from .neural_network.neural_network import NeuralNetwork


class BaseAppConfig(AppConfig):
    name = 'apps.neural_network'
    verbose_name = 'BURMOUNTE ALPHA TRADER'

    @classmethod
    def initialize_services(cls):
     #   DerivService.initialize()
     #   NeuralNetwork.initialize()
     pass

    def ready(self):
     #   self.initialize_services()
     pass

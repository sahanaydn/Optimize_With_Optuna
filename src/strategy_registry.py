from typing import Dict, Type, List
from src.strategy_base import BaseStrategy

class StrategyRegistry:
    """Strateji kayıt ve yönetim sınıfı"""
    
    # Tüm stratejileri saklayacak sözlük
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    
    @classmethod
    def register(cls, strategy_class: Type[BaseStrategy]) -> Type[BaseStrategy]:
        """
        Strateji sınıfını kaydet (decorator olarak kullanılabilir)
        
        Args:
            strategy_class: Kaydedilecek strateji sınıfı
            
        Returns:
            Type[BaseStrategy]: Kaydedilen strateji sınıfı
            
        Example:
            @StrategyRegistry.register
            class MyStrategy(BaseStrategy):
                pass
        """
        # Strateji instance'ı oluştur ve bilgilerini al
        strategy = strategy_class({})  # Boş parametrelerle
        strategy_name = strategy.strategy_info['name']
        
        # Stratejiyi kaydet
        cls._strategies[strategy_name] = strategy_class
        return strategy_class
    
    @classmethod
    def get_strategy(cls, name: str) -> Type[BaseStrategy]:
        """
        İsimle strateji sınıfını getir
        
        Args:
            name: Strateji adı
            
        Returns:
            Type[BaseStrategy]: Strateji sınıfı
            
        Raises:
            ValueError: Strateji bulunamazsa
        """
        if name not in cls._strategies:
            raise ValueError(f"Strategy '{name}' not found. Available strategies: {list(cls._strategies.keys())}")
        return cls._strategies[name]
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Kayıtlı stratejileri listele
        
        Returns:
            List[str]: Strateji adları listesi
        """
        return list(cls._strategies.keys()) 
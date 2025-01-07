import re
import json
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, TypeVar, Union, List, Callable, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

T = TypeVar("T", bound="PromoProcessor"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logs_dir = Path(__file__).resolve().parent.parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

handler = RotatingFileHandler(logs_dir / 'app.log', maxBytes=1000000, backupCount=10)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(handler)

class PromoProcessor(ABC):
    subclasses = []
    results = []
    _lock = threading.Lock()
    NUMBER_MAPPING = {"ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10}
    _store_brands = {
        'marianos': frozenset(["Private Selection", "Kroger", "Simple Truth", "Simple Truth Organic"]),
        'target': frozenset(["Deal Worthy", "Good & Gather", "Market Pantry", "Favorite Day", "Kindfull", "Smartly", "Up & Up"]),
        'jewel': frozenset(['Lucerne', "Signature Select", "O Organics", "Open Nature", "Waterfront Bistro", "Primo Taglio",
                    "Soleil", "Value Corner", "Ready Meals"]),
        'walmart': frozenset(["Clear American", "Great Value", "Home Bake Value", "Marketside", 
                    "Co Squared", "Best Occasions", "Mash-Up Coffee", "World Table"])
    }
    _compiled_patterns = {}
    _pre_processors = []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        PromoProcessor.subclasses.append(cls)
    
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        PromoProcessor.set_processor_precedence()
        self.update_save()

    @classmethod
    def apply(cls, func: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]) -> T:
        cls.results = func(cls.results)
        return cls
    
    @classmethod
    def pre_process(cls, func: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]) -> T:
        cls._pre_processors.append(func)
        return cls

    def update_save(self):
        with open("patterns.json", "w") as f:
            patterns = [pattern for subclass in self.subclasses for pattern in subclass.patterns]
            json.dump(patterns, f, indent=4)
    
    @classmethod
    @lru_cache(maxsize=1024)
    def apply_store_brands(cls, product_title: str) -> str:
        title_lower = product_title.casefold()
        for brands in cls._store_brands.values():
            if any(brand.casefold() in title_lower for brand in brands):
                return "yes"
        return "no"

    @property
    @abstractmethod
    def patterns(self):
        pass

    @abstractmethod
    def calculate_deal(self, item_data: Dict[str, Any], match: re.Match) -> Dict[str, Any]:
        pass

    @abstractmethod
    def calculate_coupon(self, item_data: Dict[str, Any], match: re.Match) -> Dict[str, Any]:
        pass

    @classmethod
    def process_item(cls, item_data: Dict[str, Any]) -> T:
        if cls._pre_processors:
            for pre_processor in cls._pre_processors:
                item_data = pre_processor(item_data)
        if isinstance(item_data, list):
            with ThreadPoolExecutor() as executor:
                processed_items = list(executor.map(cls.process_single_item, item_data))
            with cls._lock:
                cls.results.extend(processed_items)
        else:
            processed_item = cls.process_single_item(item_data)
            with cls._lock:
                cls.results.append(processed_item)
        return cls

    @classmethod
    def to_json(cls, filename: Union[str, Path]) -> None:
        if not isinstance(filename, Path):
            filename = Path(filename)
        filename = filename.with_suffix(".json") if not filename.suffix else filename
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w") as f:
            json.dump(cls.results, f, indent=4)

    @classmethod
    def _get_compiled_pattern(cls, pattern: str) -> re.Pattern:
        if pattern not in cls._compiled_patterns:
            cls._compiled_patterns[pattern] = re.compile(pattern, re.IGNORECASE)
        return cls._compiled_patterns[pattern]

    @classmethod
    def find_best_match(cls, description: str) -> Tuple[str, re.Match, "PromoProcessor", int]:
        best_score = -1
        best_pattern = None
        best_match = None
        best_processor = None
        
        for processor_class in cls.subclasses:
            processor = processor_class()
            for pattern in processor.patterns:
                compiled_pattern = cls._get_compiled_pattern(pattern)
                match = compiled_pattern.search(description)
                if match:
                    score = cls.calculate_pattern_precedence(pattern)
                    if score > best_score:
                        best_score = score
                        best_pattern = pattern
                        best_match = match
                        best_processor = processor
        
        return best_pattern, best_match, best_processor, best_score

    @classmethod
    def process_single_item(cls, item_data: Dict[str, Any]) -> Dict[str, Any]:
        updated_item = item_data.copy()
        if not hasattr(cls, "logger"):
            cls.logger = logging.getLogger(cls.__name__)
        upc = updated_item.get("upc", "")
        
        # Process deals
        deals_desc = updated_item.get("volume_deals_description", "")
        if deals_desc:
            pattern, match, processor, score = cls.find_best_match(deals_desc)
            if processor and match:
                cls.logger.info(f"UPC: {upc}: DEALS: {processor.__class__.__name__}: {deals_desc}")
                updated_item = processor.calculate_deal(updated_item, match)
                filt = lambda x: x.get("sale_price") == x.get("unit_price")
                if updated_item and filt(updated_item):
                    updated_item["volume_deals_price"] = ""
                    
        if not updated_item: return {}
        # Process coupons
        coupon_desc = updated_item.get("digital_coupon_description", "")
        if coupon_desc:
            pattern, match, processor, score = cls.find_best_match(coupon_desc)
            if processor and match:
                cls.logger.info(f"UPC: {upc}: COUPONS: {processor.__class__.__name__}: {coupon_desc}")
                updated_item = processor.calculate_coupon(updated_item, match)

        updated_item["store_brand"] = cls.apply_store_brands(updated_item["product_title"])
        return updated_item

    @staticmethod
    @lru_cache(maxsize=1024)
    def calculate_pattern_precedence(pattern: str) -> int:
        score = len(pattern) * 2
        score += len(re.findall(r'\((?!\?:).*?\)', pattern)) * 15
        score -= len(re.findall(r'[\*\+\?]', pattern)) * 8
        score -= len(re.findall(r'\{.*?\}', pattern)) * 6
        score -= len(re.findall(r'\.', pattern)) * 4
        score += len(re.findall(r'\[.*?\]', pattern)) * 5
        score += len(re.findall(r'\b', pattern)) * 3
        score += len(re.findall(r'\^|\$', pattern)) * 4
        score += len(re.findall(r'\(\?:.*?\)', pattern)) * 8
        return score

    @classmethod
    def set_processor_precedence(cls) -> None:
        for processor_class in cls.subclasses:
            processor_class.PRECEDENCE = max(
                (cls.calculate_pattern_precedence(pattern) for pattern in processor_class.patterns),
                default=0
            )

    @classmethod
    @lru_cache(maxsize=1024)
    def matcher(cls, description: str) -> str:
        pattern, match, processor, score = cls.find_best_match(description)
        return pattern
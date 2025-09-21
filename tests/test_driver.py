import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag.driver import find_revenue_driver

def test_msft_driver_found():
    res = find_revenue_driver(company="MSFT")
    assert res and isinstance(res["answer"], str) and len(res["answer"]) > 0

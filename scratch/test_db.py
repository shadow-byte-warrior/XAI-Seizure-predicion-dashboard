from src.database import init_db, save_analysis, get_history, delete_analysis

def test_db():
    print("Testing DB...")
    init_db()
    save_analysis(
        "test.edf", "DA-GRL", 0.95, 10, 100, 92.5, ["FP1", "F3"], "Test Report"
    )
    history = get_history()
    assert len(history) > 0
    print(f"History count: {len(history)}")
    print(f"Last entry: {history[0]['filename']}")
    
    # Clean up
    delete_analysis(history[0]['id'])
    history_after = get_history()
    print(f"History count after delete: {len(history_after)}")
    print("DB test passed!")

if __name__ == "__main__":
    test_db()

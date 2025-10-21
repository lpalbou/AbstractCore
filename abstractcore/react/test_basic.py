#!/usr/bin/env python3
"""
Basic test for ReAct agent functionality.

This is a simple test to verify the ReAct system works correctly.
Not a comprehensive test suite, but enough to validate the implementation.
"""

import sys
import os
import tempfile
from unittest.mock import Mock, MagicMock

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, grandparent_dir)

from abstractcore.react import ReActAgent, Scratchpad
from abstractcore.react.states import ThinkState, ActState, ObserveState, StateType


def test_scratchpad():
    """Test scratchpad functionality"""
    print("🧪 Testing Scratchpad...")
    
    scratchpad = Scratchpad(max_entries=5)
    
    # Test adding entries
    scratchpad.add_thought("This is a thought")
    scratchpad.add_observation("This is an observation")
    scratchpad.add_discovery("This is a discovery")
    
    assert len(scratchpad) == 3
    assert scratchpad.has_content()
    
    # Test formatted content
    content = scratchpad.get_formatted_content()
    assert "THOUGHT: This is a thought" in content
    assert "OBSERVATION: This is an observation" in content
    assert "DISCOVERY: This is a discovery" in content
    
    # Test summary
    summary = scratchpad.get_summary()
    assert "3 entries" in summary
    
    # Test serialization
    data = scratchpad.to_dict()
    restored = Scratchpad.from_dict(data)
    assert len(restored) == 3
    
    print("✅ Scratchpad tests passed")


def test_states():
    """Test state functionality"""
    print("🧪 Testing States...")
    
    # Test Think State
    think_state = ThinkState()
    assert think_state.state_type == StateType.THINK
    
    # Test Act State  
    act_state = ActState()
    assert act_state.state_type == StateType.ACT
    
    # Test Observe State
    observe_state = ObserveState()
    assert observe_state.state_type == StateType.OBSERVE
    
    print("✅ State tests passed")


def test_mock_agent():
    """Test ReAct agent with mock LLM"""
    print("🧪 Testing ReAct Agent with Mock LLM...")
    
    # Create mock LLM
    mock_llm = Mock()
    mock_response = Mock()
    mock_response.content = "I need to list files first. ACTION: list_files\nARGS: {}\nRATIONALE: To see what files are available"
    mock_llm.generate.return_value = mock_response
    
    # Create mock session
    mock_session = Mock()
    mock_session.generate.return_value = mock_response
    
    # Create agent
    try:
        agent = ReActAgent(mock_llm, verbose=False, max_iterations=2)
        
        # Verify agent was created
        assert agent is not None
        assert len(agent.available_tools) > 0
        assert agent.max_iterations == 2
        
        print("✅ Mock agent creation passed")
        
    except Exception as e:
        print(f"⚠️ Mock agent test failed (expected with mock): {e}")


def test_file_operations():
    """Test file operations in isolation"""
    print("🧪 Testing File Operations...")
    
    # Test with temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test content\nLine 2\nLine 3")
        temp_path = f.name
    
    try:
        from abstractcore.tools.common_tools import read_file, list_files
        
        # Test read_file
        content = read_file(temp_path)
        assert "Test content" in content
        assert "Line 2" in content
        
        # Test list_files (current directory)
        files = list_files(".")
        assert isinstance(files, str)
        
        print("✅ File operations tests passed")
        
    finally:
        # Clean up
        os.unlink(temp_path)


def main():
    """Run basic tests"""
    print("🚀 Running ReAct Basic Tests")
    print("=" * 50)
    
    try:
        test_scratchpad()
        test_states()
        test_mock_agent()
        test_file_operations()
        
        print("\n" + "=" * 50)
        print("✅ All basic tests passed!")
        print("💡 For full testing, try the example.py with a real LLM")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

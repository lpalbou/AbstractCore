"""
Example usage of structured output with AbstractLLM.
"""

from typing import List, Optional
from pydantic import BaseModel, field_validator
from abstractllm import create_llm


# Define response models
class User(BaseModel):
    name: str
    age: int
    email: str


class TaskInfo(BaseModel):
    title: str
    priority: str  # "high", "medium", "low"
    estimated_hours: Optional[float] = None
    completed: bool = False

    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v):
        if v.lower() not in ['high', 'medium', 'low']:
            raise ValueError('Priority must be high, medium, or low')
        return v.lower()


class ProductReview(BaseModel):
    product_name: str
    rating: int  # 1-5 stars
    pros: List[str]
    cons: List[str]
    would_recommend: bool

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Rating must be between 1 and 5')
        return v


def demo_mock_provider():
    """Demo with mock provider (no API keys needed)."""
    print("=== Mock Provider Demo ===")

    # Create mock LLM (no API key required)
    llm = create_llm("mock")

    # Example 1: Simple user extraction
    print("\n1. User Information Extraction:")
    try:
        user = llm.generate(
            "Extract user info: John Smith, 25 years old, john.smith@email.com",
            response_model=User
        )
        print(f"   Name: {user.name}")
        print(f"   Age: {user.age}")
        print(f"   Email: {user.email}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\nNote: Mock provider returns placeholder data for demonstration.")


def demo_openai_provider():
    """Demo with OpenAI provider (requires API key)."""
    print("\n=== OpenAI Provider Demo ===")

    try:
        # Create OpenAI LLM (requires OPENAI_API_KEY environment variable)
        llm = create_llm("openai", model="gpt-4o-mini")

        # Example 1: Task extraction with validation
        print("\n1. Task Information Extraction:")
        task = llm.generate(
            "Create a task: 'Review quarterly financial reports' - this is urgent and will take about 3 hours",
            response_model=TaskInfo
        )
        print(f"   Title: {task.title}")
        print(f"   Priority: {task.priority}")
        print(f"   Estimated Hours: {task.estimated_hours}")
        print(f"   Completed: {task.completed}")

        # Example 2: Product review extraction
        print("\n2. Product Review Extraction:")
        review_text = """
        I bought the UltraSound Pro headphones last month. Overall, I'd rate them 4 out of 5 stars.

        What I love: The sound quality is exceptional, they're very comfortable for long listening sessions,
        and the noise cancellation works great. Battery life is also impressive.

        What could be better: They're quite expensive, and the touch controls can be finicky sometimes.

        Would I recommend them? Yes, absolutely, especially for audiophiles.
        """

        review = llm.generate(
            f"Extract a structured product review from this text:\n{review_text}",
            response_model=ProductReview
        )
        print(f"   Product: {review.product_name}")
        print(f"   Rating: {review.rating}/5 stars")
        print(f"   Pros: {', '.join(review.pros)}")
        print(f"   Cons: {', '.join(review.cons)}")
        print(f"   Would Recommend: {review.would_recommend}")

    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure to set OPENAI_API_KEY environment variable")


def demo_anthropic_provider():
    """Demo with Anthropic provider (requires API key)."""
    print("\n=== Anthropic Provider Demo ===")

    try:
        # Create Anthropic LLM (requires ANTHROPIC_API_KEY environment variable)
        llm = create_llm("anthropic", model="claude-3-haiku-20240307")

        # Example: User extraction using the "tool trick"
        print("\n1. User Information Extraction (using tool trick):")
        user = llm.generate(
            "From this bio: 'Dr. Sarah Johnson is a 34-year-old researcher at MIT. Contact her at s.johnson@mit.edu'",
            response_model=User
        )
        print(f"   Name: {user.name}")
        print(f"   Age: {user.age}")
        print(f"   Email: {user.email}")

    except Exception as e:
        print(f"   Error: {e}")
        print("   Make sure to set ANTHROPIC_API_KEY environment variable")


def demo_validation_retry():
    """Demo showing validation and retry behavior."""
    print("\n=== Validation & Retry Demo ===")

    # This would show how validation errors get fed back to the LLM
    # In practice, this would use a real provider that initially fails validation
    # then succeeds on retry with feedback

    print("In real usage, if the LLM returns invalid data (e.g., priority='urgent' instead of 'high'),")
    print("the system will automatically retry with error feedback explaining the validation rules.")


if __name__ == "__main__":
    print("AbstractLLM Structured Output Examples")
    print("=" * 40)

    # Always run mock demo
    demo_mock_provider()

    # Optionally run real provider demos if API keys are available
    demo_openai_provider()
    demo_anthropic_provider()
    demo_validation_retry()

    print("\n" + "=" * 40)
    print("Examples completed!")
    print("\nTo use structured output in your code:")
    print("1. Define a Pydantic model")
    print("2. Pass it as response_model parameter to generate()")
    print("3. Get back a validated instance of your model")
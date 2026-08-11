import os
import sys
import time
import argparse
import logging

# Ensure backend directory is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.orchestration_pipeline import OrchestrationPipeline
from app.services.experience_buffer import ExperienceBufferService
from app.schemas.orchestration import OrchestrationRequest

# Suppress verbose loggers for clean script output
logging.getLogger("orchestrator").setLevel(logging.WARNING)

# 100 Diverse, High-Quality Prompts across 10 Intent Categories
PROMPT_DATASET = [
    # 1. Factual (10 prompts)
    {"category": "factual", "prompt": "What is the boiling point of water at sea level in Celsius?"},
    {"category": "factual", "prompt": "Who discovered penicillin and in which year?"},
    {"category": "factual", "prompt": "What is the chemical formula for glucose?"},
    {"category": "factual", "prompt": "Name the three largest oceans on Earth in order of area."},
    {"category": "factual", "prompt": "What is the speed of light in a vacuum in kilometers per second?"},
    {"category": "factual", "prompt": "Which element has the atomic number 79 in the periodic table?"},
    {"category": "factual", "prompt": "What is the capital of Japan and its population?"},
    {"category": "factual", "prompt": "When was the Magna Carta signed and by which King?"},
    {"category": "factual", "prompt": "What is the distance between the Earth and the Moon?"},
    {"category": "factual", "prompt": "Name the four inner terrestrial planets in our Solar System."},

    # 2. Coding (10 prompts)
    {"category": "coding", "prompt": "Write a Python function to reverse a singly linked list iteratively."},
    {"category": "coding", "prompt": "Implement a binary search algorithm in Python that returns the index of a target element."},
    {"category": "coding", "prompt": "Write a JavaScript function to throttle API requests using setTimeout."},
    {"category": "coding", "prompt": "Create a SQL query to find the top 5 highest-paid employees in each department."},
    {"category": "coding", "prompt": "Implement a LRU Cache data structure in Python with O(1) get and put operations."},
    {"category": "coding", "prompt": "Write a Python function to check if a binary tree is height-balanced."},
    {"category": "coding", "prompt": "Write a regular expression to validate email addresses according to RFC 5322."},
    {"category": "coding", "prompt": "Implement merge sort in Python and analyze its worst-case time complexity."},
    {"category": "coding", "prompt": "Write a C++ function to find the maximum subarray sum using Kadane's algorithm."},
    {"category": "coding", "prompt": "Write a Python decorator that measures and prints execution time of any function."},

    # 3. Mathematics (10 prompts)
    {"category": "mathematics", "prompt": "Calculate the derivative of f(x) = x^3 * sin(x) with respect to x."},
    {"category": "mathematics", "prompt": "Solve the quadratic equation 2x^2 + 5x - 12 = 0 for x."},
    {"category": "mathematics", "prompt": "Find the eigenvalues and eigenvectors of a 2x2 matrix [[4, 1], [2, 3]]."},
    {"category": "mathematics", "prompt": "Calculate the definite integral of x^2 from 0 to 3."},
    {"category": "mathematics", "prompt": "What is the probability of rolling two sixes with a pair of fair six-sided dice?"},
    {"category": "mathematics", "prompt": "State Bayes' Theorem and explain the meaning of prior and posterior probability."},
    {"category": "mathematics", "prompt": "Compute the Taylor series expansion for e^x centered at x = 0 up to x^4."},
    {"category": "mathematics", "prompt": "What is the dot product and cross product of vectors A = (1, 2, 3) and B = (4, 5, 6)?"},
    {"category": "mathematics", "prompt": "Solve the system of linear equations: 3x + 2y = 11 and 5x - y = 10."},
    {"category": "mathematics", "prompt": "Calculate the volume of a sphere with a radius of 7 centimeters."},

    # 4. Reasoning (10 prompts)
    {"category": "reasoning", "prompt": "Compare PostgreSQL and MongoDB for a distributed telemetry platform and recommend one based on scalability and consistency."},
    {"category": "reasoning", "prompt": "Analyze the architectural trade-offs between monolithic and microservice architectures for a financial trading application."},
    {"category": "reasoning", "prompt": "Evaluate the security and performance implications of storing JWTs in HTTP-only cookies vs browser localStorage."},
    {"category": "reasoning", "prompt": "If all A are B, and some B are C, does it logically follow that some A are C? Prove or disprove."},
    {"category": "reasoning", "prompt": "Analyze the impact of CPU cache line false sharing in multi-threaded C++ applications."},
    {"category": "reasoning", "prompt": "Compare event-driven architecture vs polling architecture for real-time stock ticker notifications."},
    {"category": "reasoning", "prompt": "What are the trade-offs of optimistic concurrency control versus pessimistic locking in high-throughput databases?"},
    {"category": "reasoning", "prompt": "Analyze the CAP Theorem in the context of Cassandra DB vs Relational Databases."},
    {"category": "reasoning", "prompt": "Evaluate the performance impact of garbage collection pauses in latency-sensitive Java applications."},
    {"category": "reasoning", "prompt": "Compare REST, GraphQL, and gRPC for client-server communication in mobile applications."},

    # 5. Summarization (10 prompts)
    {"category": "summarization", "prompt": "Summarize the key principles of the Agile Software Development Manifesto in 3 bullet points."},
    {"category": "summarization", "prompt": "Summarize the main differences between TCP and UDP protocols concisely."},
    {"category": "summarization", "prompt": "Provide a brief summary of how artificial neural networks learn via backpropagation."},
    {"category": "summarization", "prompt": "Summarize the key features and advantages of Docker containerization."},
    {"category": "summarization", "prompt": "Summarize the primary mechanisms of Git branching and merging."},
    {"category": "summarization", "prompt": "Summarize the concept of Zero Trust Security Architecture in two paragraphs."},
    {"category": "summarization", "prompt": "Summarize the main takeaways of the Raft consensus algorithm."},
    {"category": "summarization", "prompt": "Summarize the working mechanism of HTTPS and SSL/TLS handshake."},
    {"category": "summarization", "prompt": "Summarize the core concepts of Object-Oriented Programming: Encapsulation, Abstraction, Inheritance, and Polymorphism."},
    {"category": "summarization", "prompt": "Summarize the difference between synchronous and asynchronous I/O execution models."},

    # 6. Explanation (10 prompts)
    {"category": "explanation", "prompt": "Explain how virtual memory and page tables work in modern operating systems."},
    {"category": "explanation", "prompt": "Explain the concept of Garbage Collection in Python and how reference counting handles cycle detection."},
    {"category": "explanation", "prompt": "Explain how the BGE-M3 semantic embedding model generates dense vector representations."},
    {"category": "explanation", "prompt": "Explain how a B-Tree index improves database query performance."},
    {"category": "explanation", "prompt": "Explain how public-key RSA cryptography encrypts and decrypts messages."},
    {"category": "explanation", "prompt": "Explain the difference between deep learning and traditional machine learning."},
    {"category": "explanation", "prompt": "Explain how garbage collection works in the Java Virtual Machine (JVM)."},
    {"category": "explanation", "prompt": "Explain the mechanism of CORS (Cross-Origin Resource Sharing) in web browsers."},
    {"category": "explanation", "prompt": "Explain how Kubernetes manages container orchestration and pod scaling."},
    {"category": "explanation", "prompt": "Explain how a Hash Table handles hash collisions using chaining vs open addressing."},

    # 7. Creative (10 prompts)
    {"category": "creative", "prompt": "Write a short 4-line poem about an AI agent learning to navigate an environment."},
    {"category": "creative", "prompt": "Draft a creative product pitch for an AI-powered smart coffee mug."},
    {"category": "creative", "prompt": "Write a short science-fiction micro-story about a robot discovering art."},
    {"category": "creative", "prompt": "Compose a creative metaphor explaining how computer memory works to a 10-year-old."},
    {"category": "creative", "prompt": "Write a humorous haiku about debugging code at 2 AM."},
    {"category": "creative", "prompt": "Create an imaginative backstory for a legendary knight who guards a digital server room."},
    {"category": "creative", "prompt": "Draft a motivational speech for a software team starting a massive refactoring sprint."},
    {"category": "creative", "prompt": "Write a catchy tagline and 20-word description for a new quantum computing startup."},
    {"category": "creative", "prompt": "Write a dialogue between a CPU and a GPU arguing about who does more work."},
    {"category": "creative", "prompt": "Create a fictional news headline and subheadline from the year 2050 about space travel."},

    # 8. Translation (10 prompts)
    {"category": "translation", "prompt": "Translate the sentence 'Artificial intelligence is transforming software engineering' into French."},
    {"category": "translation", "prompt": "Translate 'Quality is not an act, it is a habit' into Spanish."},
    {"category": "translation", "prompt": "Translate 'Data structures and algorithms are fundamental to computer science' into German."},
    {"category": "translation", "prompt": "Translate 'Welcome to our adaptive AI platform' into Italian."},
    {"category": "translation", "prompt": "Translate 'The early bird catches the worm' into Portuguese."},
    {"category": "translation", "prompt": "Translate 'Simplicity is the soul of efficiency' into Japanese."},
    {"category": "translation", "prompt": "Translate 'Continuous integration improves code quality' into Russian."},
    {"category": "translation", "prompt": "Translate 'Knowledge is power' into Latin."},
    {"category": "translation", "prompt": "Translate 'Software is eating the world' into Dutch."},
    {"category": "translation", "prompt": "Translate 'Innovation distinguishes between a leader and a follower' into Swedish."},

    # 9. Conversational (10 prompts)
    {"category": "conversational", "prompt": "Hello! How are you doing today?"},
    {"category": "conversational", "prompt": "Can you give me a quick tip on how to stay focused while coding?"},
    {"category": "conversational", "prompt": "What is a good strategy for preparing for a technical coding interview?"},
    {"category": "conversational", "prompt": "Good morning! Can you suggest a healthy lunch idea for a busy developer?"},
    {"category": "conversational", "prompt": "What are some best practices for writing clean commit messages in Git?"},
    {"category": "conversational", "prompt": "How do you recommend balancing work and learning new tech skills?"},
    {"category": "conversational", "prompt": "Can you recommend 3 classic computer science books every engineer should read?"},
    {"category": "conversational", "prompt": "What is the best way to handle code review feedback gracefully?"},
    {"category": "conversational", "prompt": "What are some effective techniques for debugging tricky race conditions?"},
    {"category": "conversational", "prompt": "Hello! I am starting a new Python project. Any quick tips?"},

    # 10. General QA (10 prompts)
    {"category": "general_qa", "prompt": "What is the difference between a process and a thread in operating systems?"},
    {"category": "general_qa", "prompt": "How does DNS resolution work when I type a website address in a browser?"},
    {"category": "general_qa", "prompt": "What is the purpose of an API gateway in cloud architecture?"},
    {"category": "general_qa", "prompt": "What are the main components of a CPU architecture?"},
    {"category": "general_qa", "prompt": "What is the difference between static and dynamic programming languages?"},
    {"category": "general_qa", "prompt": "How does a compiler differ from an interpreter?"},
    {"category": "general_qa", "prompt": "What is the difference between latency and throughput in computer networks?"},
    {"category": "general_qa", "prompt": "What is a deadlock and what are the four coffman conditions required for it to occur?"},
    {"category": "general_qa", "prompt": "What is the purpose of database normalization up to 3NF?"},
    {"category": "general_qa", "prompt": "How does a load balancer distribute traffic across web servers?"}
]

def main():
    parser = argparse.ArgumentParser(description="Collect real Step 19 -> Step 20 experience records until target count (100) is reached.")
    parser.add_argument("--target", type=int, default=100, help="Target number of total experiences in replay buffer (default: 100)")
    parser.add_argument("--delay", type=float, default=2.0, help="Configurable delay in seconds between requests (default: 2.0 for Ollama)")
    args = parser.parse_args()

    print("=" * 80)
    print(" STANDALONE EXPERIENCE COLLECTOR (Step 19 -> Step 20 - Ollama & Multi-Provider)")
    print("=" * 80)

    # Instantiate existing ExperienceBufferService and OrchestrationPipeline
    buffer_service = ExperienceBufferService()
    pipeline = OrchestrationPipeline(experience_buffer=buffer_service)

    initial_count = buffer_service.get_status().current_size
    target_count = args.target
    remaining_needed = max(0, target_count - initial_count)

    print(f"Initial Buffer Size : {initial_count}")
    print(f"Target Buffer Size  : {target_count}")
    print(f"Remaining Needed    : {remaining_needed}")
    print(f"Inter-Request Delay : {args.delay} seconds\n")

    if initial_count >= target_count:
        print(f"Target count of {target_count} experiences is already satisfied ({initial_count} present). No collection needed.")
        return

    print("Starting collection of real experiences using existing E2E Orchestration Pipeline...\n")

    new_successful_records = 0

    for i, item in enumerate(PROMPT_DATASET):
        current_total = buffer_service.get_status().current_size
        if current_total >= target_count:
            print(f"\nTarget count of {target_count} experiences reached!")
            break

        category = item["category"]
        prompt = item["prompt"]

        print(f"[{current_total + 1}/{target_count}] Processing ({category}): \"{prompt[:60]}...\"")

        try:
            # Execute real E2E pipeline: Step 15 -> Step 16 -> Step 17 -> Step 18 -> Step 20
            res = pipeline.run_pipeline(OrchestrationRequest(prompt=prompt))

            provider_name = res.generation.provider
            selected_model = res.selected_model

            print(f"  Provider: {provider_name} | Model: {selected_model}")

            # Handle execution failure
            if not res.success or res.generation.execution_status != "completed":
                error_msg = res.generation.error_message or "Unknown execution error"
                print(f"  [ERROR] Execution failed for model '{selected_model}': {error_msg}")
                if provider_name.lower() == "google gemini" and ("429" in error_msg.lower() or "quota" in error_msg.lower()):
                    print("\n[STOP] Gemini API 429 Rate Limit / Quota Exhausted. Stopping collection cleanly.")
                    print("Persisted successful experiences remain saved in 'data/rl/experience_buffer.jsonl'.")
                    break
                else:
                    print("  Skipping failed record.")
                    continue

            # Step 20 buffer updated automatically by pipeline auxiliary hook
            latest_rec = buffer_service._buffer[-1]

            # Invariant Verification Check
            assert len(latest_rec.state) == 12, "State dimension must equal 12"
            assert latest_rec.action >= 0, "Action index must be valid"
            assert 0.0 <= latest_rec.reward <= 1.0, "Reward must be in [0, 1]"
            assert latest_rec.done is True, "done flag must be True for single-turn query"
            assert latest_rec.next_state is None, "next_state must be None"
            assert latest_rec.action_model_id == selected_model, "action_model_id must match Step 15 selected_model"

            new_successful_records += 1
            new_total = len(buffer_service._buffer)

            print(f"  -> SUCCESS! Exp ID: {latest_rec.experience_id} | Reward: {latest_rec.reward:.4f} | Buffer: {new_total}/{target_count}")

            # Inter-request delay
            if new_total < target_count:
                time.sleep(args.delay)

        except Exception as e:
            err_str = str(e)
            print(f"  [EXCEPTION] Exception occurred: {err_str}")
            if "429" in err_str.lower() or "quota" in err_str.lower():
                print("\n[STOP] Gemini API 429 Rate Limit / Quota Exhausted. Stopping collection cleanly.")
                break

    final_count = len(buffer_service._buffer)
    remaining_final = max(0, target_count - final_count)

    print("\n" + "=" * 80)
    print(" COLLECTION SUMMARY")
    print("=" * 80)
    print(f"Initial Buffer Count  : {initial_count}")
    print(f"Successful New Records: {new_successful_records}")
    print(f"Final Buffer Count    : {final_count}")
    print(f"Remaining to Target   : {remaining_final}")
    print(f"Target 100 Reached    : {final_count >= target_count}")
    print("=" * 80)

if __name__ == "__main__":
    main()

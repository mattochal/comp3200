import numpy as np

# 0 = rock, 1 = paper, 2 = scissors
scores = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])


def get_average_score(policy1, policy2, repeats=100):
    r1 = 0
    r2 = 0
    for _ in range(repeats):
        a1 = np.random.choice([0, 1, 2], p=policy1)
        a2 = np.random.choice([0, 1, 2], p=policy2)
        r1 += scores[a1][a2]
        r2 += scores[a2][a1]
    r1 /= repeats
    r2 /= repeats
    return r1, r2


def main():
    policy_param1 = np.random.randn(3)
    policy_param2 = np.random.randn(3)

    # static opponent
    policy1 = softmax(policy_param1)
    policy2 = softmax(policy_param2)

    # Go through the episode and make policy updates
    for t, transition in enumerate(episode):
        # The return after this timestep
        total_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode[t:]))
        # Calculate baseline/advantage
        baseline_value = estimator_value.predict(transition.state)
        advantage = total_return - baseline_value
        # Update our value estimator
        estimator_value.update(transition.state, total_return)
        # Update our policy estimator
        estimator_policy.update(transition.state, advantage, transition.action)

if __name__ == "__main__":
    main()
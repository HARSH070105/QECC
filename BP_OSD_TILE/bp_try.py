def normalize(prob_dict):
    # prob sum = 1
    total = sum(prob_dict.values())
    if total == 0: return prob_dict
    return {k: v / total for k, v in prob_dict.items()}

def print_step(title):
    print(f"\n{'='*40}\n{title}\n{'='*40}")


qubits = [1, 2, 3] 
# qubit labels

check_neighbors = {
    'A': [1, 3],
    'B': [2, 1],
    'C': [3, 2]
}
# define neighbours of checks

qubit_neighbors = {
    1: ['A', 'B'],
    2: ['B', 'C'],
    3: ['A', 'C']
}
# define neighbours of qubits

priors = {
    1: {'I': 0.9, 'X': 0.1},
    2: {'I': 0.9, 'X': 0.1},
    3: {'I': 0.9, 'X': 0.1}
}
# probs of error on each qubits

syndromes = {
    'A': -1, 
    'B': -1,
    'C': 1
}
# syndrome observed

m_q_to_c = {q: {c: {'I': 0.0, 'X': 0.0} for c in qubit_neighbors[q]} for q in qubits}
m_c_to_q = {c: {q: {'I': 0.0, 'X': 0.0} for q in check_neighbors[c]} for c in check_neighbors}


print_step("INITIALIZATION")

for q in qubits:
    for c in qubit_neighbors[q]:
        m_q_to_c[q][c] = priors[q].copy()
        print(f"Qubit {q} -> Check {c} initializes with Prior: {m_q_to_c[q][c]}")


for iteration in range(1, 3):
    # print_step(f"ITERATION {iteration}")
    
    # print("--- Checks process syndromes and message Qubits ---")
    for c in check_neighbors:
        q_list = check_neighbors[c]
        q_left, q_right = q_list[0], q_list[1]
        
        if syndromes[c] == 1:
            m_c_to_q[c][q_left] = m_q_to_c[q_right][c].copy()
            m_c_to_q[c][q_right] = m_q_to_c[q_left][c].copy()
        
        elif syndromes[c] == -1:
            m_c_to_q[c][q_left] = {'I': m_q_to_c[q_right][c]['X'], 'X': m_q_to_c[q_right][c]['I']}
            m_c_to_q[c][q_right] = {'I': m_q_to_c[q_left][c]['X'], 'X': m_q_to_c[q_left][c]['I']}
            
        m_c_to_q[c][q_left] = normalize(m_c_to_q[c][q_left])
        m_c_to_q[c][q_right] = normalize(m_c_to_q[c][q_right])
        
        # print(f"Check {c} (Syndrome {syndromes[c]}) -> Qubit {q_left}  : {m_c_to_q[c][q_left]}")
        # print(f"Check {c} (Syndrome {syndromes[c]}) -> Qubit {q_right}  : {m_c_to_q[c][q_right]}")

    # print("\n--- Qubits calculate current Beliefs ---")
    beliefs = {}
    for q in qubits:
        b_I = priors[q]['I']
        b_X = priors[q]['X']
        
        for c in qubit_neighbors[q]:
            b_I *= m_c_to_q[c][q]['I']
            b_X *= m_c_to_q[c][q]['X']
            
        beliefs[q] = normalize({'I': b_I, 'X': b_X})
        
        perc_I = beliefs[q]['I'] * 100
        perc_X = beliefs[q]['X'] * 100
        # print(f"Belief Qubit {q}: {perc_I:5.1f}% I, {perc_X:5.1f}% X")

    # print("\n--- Qubits update outgoing messages to Checks ---")
    for q in qubits:
        for target_c in qubit_neighbors[q]:
            m_I = priors[q]['I']
            m_X = priors[q]['X']
            
            # Multiply by messages from all OTHER checks
            for other_c in qubit_neighbors[q]:
                if other_c != target_c:
                    m_I *= m_c_to_q[other_c][q]['I']
                    m_X *= m_c_to_q[other_c][q]['X']
            
            m_q_to_c[q][target_c] = normalize({'I': m_I, 'X': m_X})
            # print(f"Qubit {q} -> Check {target_c} updated to : {m_q_to_c[q][target_c]}")

print_step("FINAL DECODING DECISION")
recovery_operation = []
for q in qubits:
    best_guess = max(beliefs[q], key=beliefs[q].get)
    recovery_operation.append(best_guess)

final_string = "".join(recovery_operation)
print(f"Highest Beliefs dictate the Recovery Operation is: [{final_string}]")
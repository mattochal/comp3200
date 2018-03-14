import torch


def update(learner_type, agent, ys, dV, delta, eta, beta):
    opponent = 1 - agent

    dVa = dV[agent]
    dVo = dV[opponent]

    n = ys[agent].size(0)  # number of states

    # Naive Learner
    if learner_type == "NL":
        return (delta * dVa[agent]).data

    # First order LOLA (assuming opponent is NL)
    if learner_type == "LOLA1":

        # 2nd order derivative of opponent's (NL's) value function w.r.t. agent's (LOLA's) parameters
        # The for-loop exists as the gradients can only be calculated from scalar values
        d2Vo_doa = [torch.autograd.grad(dVo[opponent][i], ys[agent], create_graph=True)[0] for i in range(n)]
        d2Vo_doa_Tensor = torch.cat([d2Vo_doa[i] for i in range(n)], 1)

        return (delta * dVa[agent] + delta * eta * torch.matmul(d2Vo_doa_Tensor, dVa[opponent])).data

    # First order LOLA (assuming opponent is NL) with an extra term that is left out in the origin paper
    if learner_type == "LOLA1B":

        # 2nd order derivative of opponent's (NL's) value function w.r.t. agent's (LOLA's) parameters
        # The for-loop exists as the gradients can only be calculated from scalar values
        d2Vo_doa = [torch.autograd.grad(dVo[opponent][i], ys[agent], create_graph=True)[0] for i in range(n)]
        d2Vo_doa_Tensor = torch.cat([d2Vo_doa[i] for i in range(n)], 1)

        # 2nd order derivative of agent's value function w.r.t. agent's (LOLA's) parameters
        # The for-loop exists as the gradients can only be calculated from scalar values
        d2Va_doa = [torch.autograd.grad(dVa[opponent][i], ys[agent], create_graph=True)[0] for i in range(n)]
        d2Va_doa_Tensor = torch.cat([d2Va_doa[i] for i in range(n)], 1)

        term1 = delta * dVa[agent]
        term2 = delta * eta * torch.matmul(d2Vo_doa_Tensor, dVa[opponent])
        term3 = delta * eta * torch.matmul(d2Va_doa_Tensor, dVo[opponent])
        return (term1 + term2 + term3).data

    # Second order LOLA (assuming opponent is NL)
    if learner_type == "LOLA2":

        # 2nd order derivative of opponent's (NL's) value function w.r.t. agent's (LOLA's) parameters
        d2Vo_doa = [torch.autograd.grad(dVo[opponent][i], ys[agent], create_graph=True)[0] for i in range(n)]
        d2Vo_doa_Tensor = torch.cat([d2Vo_doa[i] for i in range(n)], 1)

        d2Vo_daa = [torch.autograd.grad(dVo[agent][i], ys[agent], create_graph=True)[0] for i in range(n)]
        d2Vo_daa_Tensor = torch.cat([d2Vo_daa[i] for i in range(n)], 1)

        # Test for checking if tensor is the same as its transpose
        # print((d2Vo_daa_Tensor.data - torch.transpose(d2Vo_daa_Tensor.clone(), 0, 1).data).abs_())

        d2Va_dao = [torch.autograd.grad(dVa[agent][i], ys[opponent], create_graph=True)[0] for i in range(n)]
        d2Va_doa_Tensor = torch.cat([d2Va_dao[i] for i in range(n)], 1)

        # 3nd order derivative
        d3Va_daoa = torch.stack([torch.cat([torch.autograd.grad(d2Va_doa_Tensor[j][i], ys[agent], create_graph=True)[0]
                                            for i in range(n)], 1)
                                 for j in range(n)])

        term1 = delta * dVa[agent]
        term2 = delta * eta * torch.matmul(d2Vo_doa_Tensor, dVa[opponent])
        term3 = delta * eta * beta * torch.matmul(torch.matmul(d2Vo_daa_Tensor, d2Va_doa_Tensor), dVa[opponent])
        term4 = delta * eta * beta * torch.matmul(torch.matmul(d3Va_daoa, dVo[agent])[:, :, 0], dVa[opponent])
        return (term1 + term2 + term3 + term4).data


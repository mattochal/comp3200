import torch
from LOLA_pytorch.IPD_game import av_return
from LOLA_pytorch.IPD_modeling import policy_param_estimation_from_rollouts
from torch.autograd import Variable


def run(n=200, visualise=False):
    dtype = torch.FloatTensor

    # ######################################
    # LOLA-PG is based on trajectories
    # ######################################

    # true parameters for agent 1 and 2
    y1 = Variable(torch.zeros(5, 1).type(dtype), requires_grad=True)
    y2 = Variable(torch.zeros(5, 1).type(dtype), requires_grad=True)

    # Define rewards
    r1 = Variable(torch.Tensor([0, -3, -1, -2]).type(dtype))
    r2 = Variable(torch.Tensor([0, -1, -3, -2]).type(dtype))

    # Identity matrix
    I = Variable(torch.eye(4).type(dtype))

    # future reward discount factor
    gamma = Variable(torch.Tensor([0.8]).type(dtype))

    # Term in f_nl update rule
    delta = Variable(torch.Tensor([0.1]).type(dtype))

    # Term in f_lola update rule
    eta = Variable(torch.Tensor([10]).type(dtype))

    for epoch in range(n):
        theta1 = torch.sigmoid(y1)
        theta2 = torch.sigmoid(y2)

        # These are the exact value functions of the agent 1 and 2
        pm1Y, pm2Y = policy_param_estimation_from_rollouts(theta1, theta2)
        pm1Y = Variable(torch.from_numpy(pm1Y).float(), requires_grad=True)
        pm2Y = Variable(torch.from_numpy(pm2Y).float(), requires_grad=True)

        # Sigmoid function is to ensure that the parameters are within 1
        m_theta1 = torch.sigmoid(pm1Y)
        m_theta2 = torch.sigmoid(pm2Y)

        # Agent 1 knows own policy, and models agent 2's policy
        P1 = torch.cat((theta1 * m_theta2, theta1 * (1 - m_theta2), (1 - theta1) * m_theta2, (1 - theta1) * (1 - m_theta2)), 1)

        # Agent 2 knows its own policy, and models agent 1's policy
        P2 = torch.cat((m_theta1 * theta2, m_theta1 * (1 - theta2), (1 - m_theta1) * theta2, (1 - m_theta1) * (1 - theta2)), 1)

        Zinv1 = torch.inverse(I - gamma * P1[1:, :])
        Zinv2 = torch.inverse(I - gamma * P2[1:, :])

        V1 = torch.matmul(torch.matmul(P1[0, :], Zinv1), r1)  # True value function of agent 1
        V2 = torch.matmul(torch.matmul(P2[0, :], Zinv2), r2)  # True value function of agent 2
        V2_from1 = torch.matmul(torch.matmul(P1[0, :], Zinv1), r2)  # V2 from agent 1's perspective
        V1_from2 = torch.matmul(torch.matmul(P2[0, :], Zinv2), r1)  # V1 from agent 2's perspective

        # 1st order gradient of 1st and 2nd agents' value functions w.r.t. both agents' parameters
        # Note: even though we generate the derivative with respect to both agents' policies
        #       later we only take the partial derivative with respect to agent's own policy
        dV1 = torch.autograd.grad(V1, (y1, pm2Y), create_graph=True)
        dV2 = torch.autograd.grad(V2, (pm1Y, y2), create_graph=True)

        # 1st order derivative of value functions but from each agents' perspectives
        dV21 = torch.autograd.grad(V2_from1, pm2Y, create_graph=True)  # Why?! Is this the difference between PG and OM?
        dV12 = torch.autograd.grad(V1_from2, pm1Y, create_graph=True)

        # 2nd order gradient of 1st agent's value function w.r.t. 2nd agents' parameters
        # The for-loop exists as the gradients can only be calculated from scalar values
        d2V2 = [torch.autograd.grad(dV21[0][i], y1, create_graph=True)[0] for i in range(y1.size(0))]
        d2V1 = [torch.autograd.grad(dV12[0][i], y2, create_graph=True)[0] for i in range(y1.size(0))]
        d2V2Tensor = torch.cat([d2V2[i] for i in range(y1.size(0))], 1)
        d2V1Tensor = torch.cat([d2V1[i] for i in range(y1.size(0))], 1)

        # Update for LOLA agent
        y1.data += (delta * dV1[0] + delta * eta * torch.matmul(d2V2Tensor, dV1[1])).data

        # Update for LOLA agent
        y2.data += (delta * dV2[1] + delta * eta * torch.matmul(d2V1Tensor, dV2[0])).data

        if epoch % 10 == 0:
            print("Epoch {}".format(epoch))
            if visualise:
                print("x1: {}".format(theta1.data.cpu().numpy().tolist()))
                print("V1: {}".format(V1.data[0]))
                print("x2: {}".format(theta2.data.cpu().numpy().tolist()))
                print("V2: {}".format(V2.data[0]))
                print("Rewards: {}".format(av_return(theta1, theta2)))

    # return policy of both agents
    return torch.sigmoid(y1), torch.sigmoid(y2)

if __name__ == "__main__":
    run(visualise=True)
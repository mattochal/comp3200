import numpy as np
import torch
from myLOLA.update_rules import update
from torch.autograd import Variable

from myLOLA.av_return import av_return
from myLOLA.value_fns import exact, policy_grad, modelling, full_modelling
from agent_pair import AgentPair

run_signature = ["length", "visualise", "record_trajectory", "payoff1", "payoff2", "gamma", "delta", "eta", "beta",
                 "init_policy1", "init_policy2", "rollout_length", "num_rollout", "agents", "value_fn", "exact"]


def run(length=200, visualise=False, record_trajectory=False, payoff1=[-1, -3, 0, -2], payoff2=[-1, 0, -3, -2],
        gamma=0.8, delta=0.1, eta=10, beta=0, num_rollout=50, rollout_length=50,
        agents=["LOLA1", "LOLA1"], value_fn="exact",
        init_policy1=[0.5, 0.5, 0.5, 0.5, 0.5], init_policy2=[0.5, 0.5, 0.5, 0.5, 0.5]):
    dtype = torch.FloatTensor
    results = {"epoch": []}

    init_policy1 = np.array(init_policy1, dtype="f")
    init_policy2 = np.array(init_policy2, dtype="f")

    # policy parameters
    y1 = np.log(np.divide(init_policy1, 1 - init_policy1)).reshape((5, 1))
    y2 = np.log(np.divide(init_policy2, 1 - init_policy2)).reshape((5, 1))
    y1 = Variable(torch.from_numpy(y1).float(), requires_grad=True)
    y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True)

    # modeled parameters, initially assumed 50% chance of cooperation in all states
    my1 = Variable(torch.Tensor([0, 0, 0, 0, 0]).type(dtype), requires_grad=True)
    my2 = Variable(torch.Tensor([0, 0, 0, 0, 0]).type(dtype), requires_grad=True)

    # Define rewards
    r1 = Variable(torch.Tensor(payoff1).type(dtype))
    r2 = Variable(torch.Tensor(payoff2).type(dtype))

    # future reward discount factor
    gamma = Variable(torch.Tensor([gamma]).type(dtype))

    # Term in f_nl update rule
    delta = Variable(torch.Tensor([delta]).type(dtype))

    # Term in f_lola1 update rule (first order LOLA)
    eta = Variable(torch.Tensor([eta]).type(dtype))

    # Term in f_lola2 update rule (higher order LOLA)
    beta = Variable(torch.Tensor([beta]).type(dtype))

    for epoch in range(length):
        x1 = torch.sigmoid(y1)
        x2 = torch.sigmoid(y2)

        if value_fn == "exact":
            V1, V2, dV1, dV2 = exact(y1, y2, r1, r2, gamma)
            dy1 = update(agents[0], 0, (y1, y2), (dV1, dV2), delta, eta, beta)
            dy2 = update(agents[1], 1, (y1, y2), (dV1, dV2), delta, eta, beta)

            # print("dy1: ", dy1.cpu().numpy())
            # print("dy2: ", dy1.cpu().numpy())

            y1.data += dy1
            y2.data += dy2

        elif value_fn == "policy_grad":
            V1, V2, dV1, dV2 = exact(y1, y2, r1, r2, gamma)
            _, _, dV1_from2, dV2_from1 = policy_grad(y1, y2, r1, r2, gamma)
            dy1 = update(agents[0], 0, (y1, y2), (dV1, dV2_from1), delta, eta, beta)
            dy2 = update(agents[1], 1, (y1, y2), (dV1_from2, dV2), delta, eta, beta)
            y1.data += dy1
            y2.data += dy2

        # Modeling of the opponent parameters only
        elif value_fn == "modelling":
            V1, V2, dV1, dV2 = exact(y1, y2, r1, r2, gamma)

            # dV1_from1, dV2_from1 are gradients of the value functions from perspective of agent 1
            # similarly is the case for agent 2
            _, _, dV1_from1, dV2_from1, dV1_from2, dV2_from2, my1, my2 = \
                modelling(y1, y2, r1, r2, gamma, rollout_length, num_rollout, my1, my2)

            dy1 = update(agents[0], 0, (y1, my2), (dV1_from1, dV2_from1), delta, eta, beta)
            dy2 = update(agents[1], 1, (my1, y2), (dV1_from2, dV2_from2), delta, eta, beta)
            y1.data += dy1
            y2.data += dy2

        # Modeling of the value function and parameters,
        # i.e. using policy gradients to model gradient of value function and modeling the policy parameters
        elif value_fn == "full_modelling":
            V1, V2, dV1, dV2 = exact(y1, y2, r1, r2, gamma)

            # dV1_from1, dV2_from1 are gradients of the value functions from perspective of agent 1
            _, _, dV1_from1, dV2_from1, dV1_from2, dV2_from2, my1, my2 = \
                full_modelling(y1, y2, r1, r2, gamma, rollout_length, num_rollout, my1, my2)

            dy1 = update(agents[0], 0, (y1, my2), (dV1_from1, dV2_from1), delta, eta, beta)
            dy2 = update(agents[1], 1, (my1, y2), (dV1_from2, dV2_from2), delta, eta, beta)
            y1.data += dy1
            y2.data += dy2

        if record_trajectory:
            results["epoch"].append({"P1": np.squeeze(x1.data.cpu().numpy()).tolist(),
                                     "P2": np.squeeze(x2.data.cpu().numpy()).tolist()})

        # stft1 = np.abs(torch.sigmoid(y1).data.cpu().numpy() - [[1], [1], [0], [1], [0]])
        # stft2 = np.abs(torch.sigmoid(y2).data.cpu().numpy() - [[1], [1], [1], [0], [0]])
        # print("TFT new x1: ", np.all(stft1 < 0.25), sum(stft1))
        # print("TFT new x2: ", np.all(stft2 < 0.25), sum(stft2))
        # print("diff x1: ", (torch.sigmoid(y1) - x1).data.cpu().numpy())
        # print("diff x2: ", (torch.sigmoid(y2) - x2).data.cpu().numpy())

        if epoch % 20 == 0 and visualise:
            print('Epoch: ' + str(epoch))
            print("x1: {}".format(x1.data.cpu().numpy().tolist()))
            print("V1: {}".format(V1.data[0]))
            print("x2: {}".format(x2.data.cpu().numpy().tolist()))
            print("V2: {}".format(V2.data[0]))
            print("Rewards: {}".format(av_return(x1, x2)))

    # return policy of both agents
    # print("init x1: {}".format(init_policy1))
    # print("init x2: {}".format(init_policy2))
    # print("x1: {}".format(x1.data.cpu().numpy().tolist()))
    # print("x2: {}".format(x2.data.cpu().numpy().tolist()))
    p1, p2 = torch.sigmoid(y1), torch.sigmoid(y2),
    results["P1"] = np.squeeze(p1.data.cpu().numpy()).tolist()
    results["P2"] = np.squeeze(p2.data.cpu().numpy()).tolist()
    return p1, p2, results


class AgentPairExact(AgentPair):
    def run(self, seed):
        super(AgentPairExact, self).run(seed)
        self.parameters["value_fn"] = "exact"
        self.parameters["record_trajectory"] = True
        return run(**self.parameters)


class NL_VS_NL(AgentPairExact):
    def run(self, seed):
        self.parameters["agents"] = ["NL", "NL"]
        return super(NL_VS_NL, self).run(seed)


class LOLA1_VS_LOLA1(AgentPairExact):
    def run(self, seed):
        self.parameters["agents"] = ["LOLA1", "LOLA1"]
        return super(LOLA1_VS_LOLA1, self).run(seed)


class LOLA1B_VS_LOLA1B(AgentPairExact):
    def run(self, seed):
        self.parameters["agents"] = ["LOLA1B", "LOLA1B"]
        return super(LOLA1B_VS_LOLA1B, self).run(seed)


class LOLA1_VS_NL(AgentPairExact):
    def run(self, seed):
        self.parameters["agents"] = ["LOLA1", "NL"]
        return super(LOLA1_VS_NL, self).run(seed)


class LOLA1B_VS_NL(AgentPairExact):
    def run(self, seed):
        self.parameters["agents"] = ["LOLA1B", "NL"]
        return super(LOLA1B_VS_NL, self).run(seed)


class LOLA1B_VS_LOLA1(AgentPairExact):
    def run(self, seed):
        self.parameters["agents"] = ["LOLA1B", "NL"]
        return super(LOLA1B_VS_LOLA1, self).run(seed)


class LOLA1_VS_LOLA1_PG(AgentPair):
    def run(self, seed):
        super(LOLA1_VS_LOLA1_PG, self).run(seed)
        self.parameters["agents"] = ["LOLA1", "LOLA1"]
        self.parameters["value_fn"] = "policy_grad"
        # self.parameters["record_trajectory"] = False
        return run(**self.parameters)


class LOLA1_VS_LOLA1_OM(AgentPair):
    def run(self, seed):
        super(LOLA1_VS_LOLA1_OM, self).run(seed)
        self.parameters["agents"] = ["LOLA1", "LOLA1"]
        self.parameters["value_fn"] = "modelling"
        return run(**self.parameters)


if __name__ == "__main__":
    run(visualise=True)

#
# import torch
# from torch.autograd import Variable
# import numpy as np
#
# dtype = torch.FloatTensor
# init_policy1 = np.array([0.75, 0.95, 0.45, 0.53, 0.54], dtype="f")
# init_policy2 = np.array([0.35, 0.25, 0.75, 0.35, 0.55], dtype="f")
#
# y1 = np.log(np.divide(init_policy1, 1 - init_policy1)).reshape((5, 1))
# y1 = Variable(torch.from_numpy(y1).float(), requires_grad=True)
# x1 = torch.sigmoid(y1)
#
# y2 = np.log(np.divide(init_policy2, 1 - init_policy2)).reshape((5, 1))
# y2 = Variable(torch.from_numpy(y2).float(), requires_grad=True)
# x2 = torch.sigmoid(y2)
#
# I = Variable(torch.eye(4).type(dtype))
# gamma = Variable(torch.Tensor([0.97]).type(dtype))
#
# P = torch.cat((x1 * x2, x1 * (1 - x2), (1 - x1) * x2, (1 - x1) * (1 - x2)), 1)
# Zinv = torch.inverse(I - gamma * P[1:, :])
# r1 = Variable(torch.Tensor([-1, -3, 0, -2]).type(dtype))
# r2 = Variable(torch.Tensor([-1, 0, -3, -2]).type(dtype))
#
# V1 = torch.matmul(torch.matmul(P[0, :], Zinv), r1)
# V2 = torch.matmul(torch.matmul(P[0, :], Zinv), r2)
# dV1 = torch.autograd.grad(V1, (y1, y2), create_graph=True)
# dV2 = torch.autograd.grad(V1, (y1, y2), create_graph=True)
#
# dV2_d2 = dV2[1]
# d2V2_d12 = [torch.autograd.grad(dV2_d2[i], y1, create_graph=True)[0] for i in range(5)]
# d2V2_d12_Tensor = torch.cat([d2V2_d12[i] for i in range(5)], 1)

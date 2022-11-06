import torch

def make_postprocessor_mlp(Y_pred, Y):  # expects Y_pred and Y at subject level
    model = torch.nn.Linear(1,1)
    optim = torch.optim.SGD(model.parameters(), lr=.1)

    loss_fun = torch.nn.L1Loss()

    X = (torch.from_numpy(Y_pred.reshape(-1,1)) - 50) / 50
    y = (torch.from_numpy(Y) - 50) / 50

    error = list()
    model.train()
    for epoch in range(1000):
        optim.zero_grad()

        y_pred = model(X).squeeze()

        loss = loss_fun(y_pred, y)

        loss.backward()
        optim.step()

        error.append(loss.item())
    
    model.eval()
    return lambda y: model((torch.from_numpy(y.reshape(-1,1)) - 50) / 50).squeeze().detach().numpy() * 50 + 50
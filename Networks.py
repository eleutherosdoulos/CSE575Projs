import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        #note: my general impression is more neurons in a layer yield better fitting
        #and more layers yield better generalizability. I'm not getting great fitting
        #even on train data, so, let's go wide before deep
        input_size = 6 #don't change
        hidden1_size = 500
        hidden2_size = 400
        hidden3_size = 300
        final_hidden_size = 200
        output_size = 1 #don't change
        super(Action_Conditioned_FF, self).__init__() 
        self.input_to_hidden1 = nn.Linear(input_size, hidden1_size)
        self.nonlinear_activation = nn.ReLU()
        #any additional layers below here
        self.hidden1_to_hidden2 = nn.Linear(hidden1_size,hidden2_size)
        self.hidden2_to_hidden3 = nn.Linear(hidden2_size,hidden3_size)
        #end of additional layers
        self.hidden_to_final_hidden = nn.Linear(hidden3_size,final_hidden_size)
        self.fc_layer = nn.Linear(final_hidden_size, output_size)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden1(input.float())
        hidden = self.nonlinear_activation(hidden)
        #any additional layers below here
        hidden = self.hidden1_to_hidden2(hidden)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.hidden2_to_hidden3(hidden)
        hidden = self.nonlinear_activation(hidden)
        #end of additional layers
        hidden = self.hidden_to_final_hidden(hidden)
        hidden = self.nonlinear_activation(hidden)
        output = self.fc_layer(hidden)  
        return output
    
    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.

        losses = [] 

        for batch in test_loader:
            output = torch.flatten(model.forward(batch['input'])).float()
            target = torch.flatten(batch['label']).float()
            batch_loss = loss_function(output, target)
            losses.append(batch_loss.item())
            loss = sum(losses)/len(losses)
        return loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()

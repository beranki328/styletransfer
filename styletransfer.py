"""Define two distances, one for measuring the difference in content(Dc),
and one for measuring the difference in style between the two imgs(Ds)
take input as third image and transform it to minimize the difference
between content and style w/ Dc and Ds
this process transfers the style from the Dc and Ds into a third input pic, thus STYLE TRANSFER.
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

#need to define UTILITY FUNCTIONS:
#function turns an image into a tensor to be used as an input for the model
def load_image(img_path, max_size = 400, shape = None):
    image = Image.open(img_path).convert(0)
    
    if max(image.size) > max_size:
        size = max_size
    if shape is not None:
        size = shape  
    in_transform = transforms.Compose([transforms.Resize((size, int(1.5 * size))), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

style = load_image("starrynight.jpg")
style.shape

#function returns tensor w shape torch.Size([1, 3, 400, 600])
#need to define a function that can do the opposite of that(rearranges)
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    
    return image

#following functions performs a forward pass thru model, on a one layer basis
#then stores the fature map if the name of the layer matches a name in predef. layer
#predef. layer dict serves as mapping for VG19 implementation's layer indices
#to the layer names defined in the paper; else, complete set of content and style layers used by def.
    
def get_features(image, model, layers = None):
    if layers is None:
       layers = {'0': 'conv1_1','5': 'conv2_1',
                 '10': 'conv3_1',
                 '19': 'conv4_1',
                 '21': 'conv4_2',  ## content layer
                 '28': 'conv5_1'}
       features = {}
       x = image
       for name, layer in enumerate(model.features):
           x = layer(x)
           if str(name) in layers:
               features[layers[str(name)]] = x
               
       return features

#need to focus on the represntation of style in layers; can be obtained by
#correlations between diff. feature maps responses of a given layer
#boils down to computing the Gram matrix of the vectorized ft. map
#take a ft. map tensor as input and reshape the extents of the tensor to 1 vct.
#compute inner product of reshaped tensor
       
def gram_matrix(tensor):
    _, n_filters, h, w = tensor.size()
    tensor = tensor.view(n_filters, h * w)
    gram = torch.mm(tensor, tensor.t())
    
    return gram

#load the VGG model with pre-trained weights; sets requires_grad to False
#so that no gradients r computed for model's weights
    
torch.utils.model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', model_dir='/results/')
vgg = models.vgg19()
vgg.load_state_dict(torch.load('vgg19-dcbb9e9d.pth'))

for param in vgg.parameters():
    param.requires_grad_(False)

for i, layer in enumerate(vgg.features):
    if isinstance(layer, torch.nn.MaxPool2d):
        vgg.features[i] = torch.nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        
torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device).eval()


#now loading in content image to compute the feature map responses
content = load_image('shanghai.jpg').to(device)
style = style.to(device)

cf = get_features(content, vgg)
sf = get_features(style, vgg)

#computing gram matrices for all style ft. layers and putting them in dict.
sg = {layer: gram_matrix(sf[layer]) for layer in sf}

#need to make 3rd image to start image trans. process using orig. content img
target = torch.randn_like(content).requires_grad_(True).to(device)

#need to have multiple style layers contributing, each w their own importance
style_weights = {'conv1_1': 0.75, 'conv2_1': 0.5, 'conv3_1': 0.25, 'conv4_1': 0.2, 'conv5_1': 0.2}

#want weights for the overall strength of both loss terms(Dc and Ds)
cw = 1e4
sw = 1e2

#constructing total loss for style transfer; similar to
#construct of total content transfer but replaces ft. map responses w gram matrices
#and some tweaks in the loss function

optim = optim.Adam([target], lr = 0.01)

#need to define the number of iters for the style transfer loop using
#content and style losses, multiplying them w respective weights, and + them 
#for total results; then backpropogate and update pixel vals of image till done

for i in range(1, 401):
    optim.zero_grad()
    tf= get_features(target, vgg)
    content_loss = torch.mean((tf['conv4_2'] - cf['conv4_2']) ** 2)
    
    style_loss = 0
    for layer in style_weights:
        target_feature = tf[layer]
        target_gram = gram_matrix(target_feature)
    _, d, h, w = target_feature.shape
    style_g = sg[layer]
    layer_style_loss = style_weights[layer] * torch.mean(
      (target_gram - style_g) ** 2)
    style_loss += layer_style_loss / (d * h * w)
    
    total_loss = cw * content_loss + sw * style_loss
    total_loss.backward(retain_graph=True)
    optim.step()
    
    if i % 50 == 0:
        total_loss_rounded = round(total_loss.item(), 2)
        content_fraction = round(
            cw *content_loss.item()/total_loss.item(), 2)
        style_fraction = round(
            sw * style_loss.item()/total_loss.item(), 2)
        print('Iteration {}, Total loss: {} - (content: {}, style {})'.format(
            i,total_loss_rounded, content_fraction, style_fraction))
      
final_img = im_convert(target)
    
fig = plt.figure()
plt.imshow(final_img)
plt.axis('off')
plt.savefig('/results/modern_starry.png')

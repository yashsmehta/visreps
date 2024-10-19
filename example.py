def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks for convolutional layers
model.features[0].register_forward_hook(get_activation('conv1'))
model.features[3].register_forward_hook(get_activation('conv2'))
model.features[6].register_forward_hook(get_activation('conv3'))
model.features[8].register_forward_hook(get_activation('conv4'))
model.features[10].register_forward_hook(get_activation('conv5'))

# Register hooks for fully connected layers
model.classifier[1].register_forward_hook(get_activation('fc6'))
model.classifier[4].register_forward_hook(get_activation('fc7'))
model.classifier[6].register_forward_hook(get_activation('fc8'))

# ... rest of your code ...

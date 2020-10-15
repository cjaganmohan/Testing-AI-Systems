# reference: https://towardsdatascience.com/extract-features-visualize-filters-and-feature-maps-in-vgg16-and-vgg19-cnn-models-d2da6333edd0
import numpy as np
from matplotlib import pyplot
from numpy import expand_dims
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# load the model
model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=model.inputs, outputs=model.get_layer('block5_pool').output)
model.summary()


# image
img_src = '1479425660620933516.jpg'
img = image.load_img(img_src, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)



#
# # image
# img_src = '1479425660620933516_Blur_Avg_3.jpg'
# img = image.load_img(img_src, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# features_Blur_Avg_3 = model.predict(x)
#
#
# # image
# img_src = '1479425660620933516_Blur_Avg_6.jpg'
# img = image.load_img(img_src, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# features_Blur_Avg_6 = model.predict(x)
#
# # image
# img_src = '1479425660620933516_Contrast_1.8.jpg'
# img = image.load_img(img_src, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# features_Contrast_1_8 = model.predict(x)
#
# # image
# img_src = '1479425660620933516_Contrast_2.2.jpg'
# img = image.load_img(img_src, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# features_Contrast_2_2 = model.predict(x)
#
# # image
# img_src = '1479425660620933516_Translation_10.jpg'
# img = image.load_img(img_src, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# features_Translation_10 = model.predict(x)
#
#
#
# # print(features.flatten().shape)
# # print(features_Blur_Avg_3.flatten().shape)
#
# features_1d = features.flatten()
# features_1_1d = features_Blur_Avg_3.flatten()
#
# # print(features_1d.shape)
# # print(features_1d.shape)
# #
# # print(features_1d.reshape(-1,2).shape)
# # print(features_1_1d.reshape(-1,2).shape)
#
# #cos_similarity = cosine_similarity(features_1d.reshape(-1,2), features_1_1d.reshape(-1,2))
#
# # cos_similarity = distance.cosine(features_1d, features_1_1d)
# # cos_similarity = distance.cosine(features_1d, features_Blur_Avg_6.flatten())
# # cos_similarity = distance.cosine(features_1d, features_Contrast_1_8.flatten())
# # cos_similarity = distance.cosine(features_1d, features_Contrast_2_2.flatten())
#
# # print(cos_similarity)
# # print(type(cos_similarity))
#
# print(distance.cosine(features_1d, features_1_1d))
# print(distance.cosine(features_1d, features_Blur_Avg_6.flatten()))
# print(distance.cosine(features_1d, features_Contrast_1_8.flatten()))
# print(distance.cosine(features_1d, features_Contrast_2_2.flatten()))
# print(distance.cosine(features_1d, features_Translation_10.flatten()))




# load the model
#model = VGG16(weights='imagenet', include_top=False)
#model = VGG16(weights='imagenet')
model = VGG16()
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs,outputs=outputs)

# model = Model(inputs=model.inputs, outputs=model.get_layer('block4_pool').output)
# model.summary()

# image
img_src = '../ImageTransformations/1479425441182877835.jpg'
img = image.load_img(img_src, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

feature_maps = model.predict(x)
#print(features.shape)
#print(features)
#
square = 8


for fmap in feature_maps:
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    pyplot.show('First layer')

# print("-------Now with block 4------------")
#
#
# # load the model
# #model = VGG16(weights='imagenet', include_top=False)
# model = VGG16(weights='imagenet')
# #model = VGG16()
# model = Model(inputs=model.inputs, outputs=model.get_layer('block4_pool').output)
# model.summary()
#
# # image
# img_src = '../ImageTransformations/1479425441182877835.jpg'
# img = image.load_img(img_src, target_size=(224, 224))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
#
# features = model.predict(x)
# print(features.shape)
# #print(features)
# #
# square = 8
# ix = 1
#
# for _ in range(square):
#     for _ in range(square):
#         ax = pyplot.subplot(square, square, ix)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         pyplot.imshow(features[0, :, :, ix-1], cmap='gray')
#         ix += 1
# pyplot.show('Block_4')


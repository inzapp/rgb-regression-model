# RGB Regression Model

This model is a regression model that predicts RGB values for ROI from images.

There are many ways to extract colors for ROI from images.

In a very simple way, you take the average of the entire color of the image.

However, this method has the disadvantage that the smaller the area you are trying to extract the color, the less accurate it is.

In addition, the color of the background area and the area you want to extract the color are different, the accuracy decreases.

You may want a more precise way.

You can create a mask that extracts only the desired part of the image using the segmentation model or detection model, and there is a way to average the color of the mask area.

However, this method also requires additional crop & resize operations and may require high throughput for segmentation models.

I suggest a very simple and effective way.

It's about learning the colors of the ROI directly to the CNN model.

Even though there was no information about the ROI area, I knew it worked very well.

No additional crop & resize operations are required

And you don't need a huge size model.

You only need one light model to extract color from an image.

## Experiment

The output of this model is the RGB color code value for the ROI in the image.

But can CNN really extract the color of the image?

I found out through a simple experiment that this is possible.

<img src="/md/rgb.jpg" width="500"><br>

Each image on the left is an input image, and the image on the right is the result of converting RGB values from the output of the regression model into an image.

The model used for the training is a 128x128x3 model and trained with approximately 1 million parameters.

<img src="/md/rgb_noise.jpg" width="500"><br>

The above image shows that colors can also be extracted for the ROI from images that contain noise.

This means that you can actually use this model.

## Labeling

If so, what labeling tools should be used to create training data?

I created a very convenient labeling tool based on OpenCV.

All you have to do is copy the label_rgb.py file to where the images are located and run it.

<img src="/md/label_rgb.gif" width="500"><br>

You can label the color of the desired coordinate in the image as the representative color.

Use the color bar above if you want a more chromatic color.

The color bar can be replaced with any color through simple code modification.

```python
def add_color_table_on_top(view):
    ...
    color_table = [
        # modify here
    ]
    ...
```

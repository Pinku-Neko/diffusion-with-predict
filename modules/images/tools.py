from matplotlib import pyplot as plt
import matplotlib.animation as animation


def plot_grayscale_image(image):
    plt.imshow(image,cmap='gray')
    plt.show()


def animate_grayscale_images(images):
    fig = plt.figure()
    ims = []
    for i in range(len(images)):
        im = plt.imshow(images[i], cmap="gray", animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()

    import matplotlib.pyplot as plt
from matplotlib import animation

def compare_grayscale_animations_2(images_long, images_short):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a subplot with two axes (side by side)
    
    # Initialize two lists to store the animations
    ims_long = []
    ims_short = []
    frame_diff = len(images_long) - len(images_short)

    # Create animations for the first set of images
    for i in range(len(images_long)):
        im_long = axes[0].imshow(images_long[i], cmap="gray", animated=True)
        ims_long.append([im_long])
        # if short animation does not start
        if i >= frame_diff:
            # use i + offset as index to grab image
            im_short = axes[1].imshow(images_short[i-frame_diff], cmap="gray", animated=True)
        else:
            # feed 1st image
            im_short = axes[1].imshow(images_short[0], cmap="gray", animated=True)
        ims_short.append([im_short])

    # Create ArtistAnimations for both sets of images
    animate1 = animation.ArtistAnimation(fig, ims_long, interval=50, blit=True, repeat_delay=1000)
    animate2 = animation.ArtistAnimation(fig, ims_short, interval=50, blit=True, repeat_delay=1000)


    plt.show()


def compare_grayscale_animations(images_long, images_short):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a subplot with two axes (side by side)
    
    # Initialize two lists to store the animations
    ims = []
    frame_diff = len(images_long) - len(images_short)

    # Create animations for the first set of images
    for i in range(len(images_long)):
        im_long = axes[0].imshow(images_long[i], cmap="gray", animated=True)
        # if short animation does not start
        if i >= frame_diff:
            # use i + offset as index to grab image
            im_short = axes[1].imshow(images_short[i-frame_diff], cmap="gray", animated=True)
        else:
            # feed 1st image
            im_short = axes[1].imshow(images_short[0], cmap="gray", animated=True)
        ims.append([im_long,im_short])

    # repeat last frame to observe result
    last_frame_long = axes[0].imshow(images_long[-1], cmap="gray", animated=True)
    last_frame_short = axes[1].imshow(images_short[-1], cmap="gray", animated=True)
    for i in range(50):
        ims.append([last_frame_long,last_frame_short])

    # Create ArtistAnimations for both sets of images
    animate = animation.ArtistAnimation(fig, ims, interval=25, blit=True, repeat_delay=1000)
    animate.save("comparison.gif")

    plt.show()
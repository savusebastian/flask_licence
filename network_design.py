import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(facecolor='w')
ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[])
plt.box(False)
circ = plt.Circle((1, 1), 2)

radius = 0.3

arrow_kwargs = dict(head_width=0.05, fc='black')


# =============================
# Function to draw arrows
# =============================
def draw_connecting_arrow(ax, circ1, rad1, circ2, rad2):
    theta = np.arctan2(circ2[1] - circ1[1], circ2[0] - circ1[0])
    starting_point = (circ1[0] + rad1 * np.cos(theta), circ1[1] + rad1 * np.sin(theta))
    length = (circ2[0] - circ1[0] - (rad1 + 1.4 * rad2) * np.cos(theta), circ2[1] - circ1[1] - (rad1 + 1.4 * rad2) * np.sin(theta))
    ax.arrow(starting_point[0], starting_point[1], length[0], length[1], **arrow_kwargs)


# =============================
# Function to draw circles
# =============================
def draw_circle(ax, center, radius):
    circ = plt.Circle(center, radius, fc='#afc1c4', ec='purple', lw=2)
    ax.add_patch(circ)


x1 = -6
x2 = -3
x3 = -1
x4 = 1
x5 = 3
x6 = 6
y6 = 6

# =============================
# Draw circles
# =============================
# for i, y1 in enumerate(np.linspace(1.5, -1.5, 4)):
#     draw_circle(ax, (x1, y1), radius)
#     ax.text(x1 - 0.9, y1, 'Input #%i' % (i + 1), ha='right', va='center', fontsize=14)
#     draw_connecting_arrow(ax, (x1 - 0.9, y1), 0.1, (x1, y1), radius)

draw_circle(ax, (x1, 0), radius)
ax.text(x1 - 0.9, 0, 'Input', ha='right', va='center', fontsize=14)
draw_connecting_arrow(ax, (x1 - 0.9, 0), 0.1, (x1, 0), radius)

for y2 in np.linspace(-2, 2, 4):
    draw_circle(ax, (x2, y2), radius)

for y3 in np.linspace(-2, 2, 4):
    draw_circle(ax, (x3, y3), radius)

for y4 in np.linspace(-2, 2, 4):
    draw_circle(ax, (x4, y4), radius)

for y5 in np.linspace(-2, 2, 4):
    draw_circle(ax, (x5, y5), radius)

draw_circle(ax, (x6, 0), radius)
ax.text(x6 + 0.8, 0, 'Output', ha='left', va='center', fontsize=14)
draw_connecting_arrow(ax, (x6, y6), radius, (x6 + 0.8, y6), 0.1)

# =============================
# Draw connecting arrows
# =============================
for y2 in np.linspace(-2, 2, 4):
    draw_connecting_arrow(ax, (x1, 0), radius, (x2, y2), radius)

for y2 in np.linspace(-2, 2, 4):
    for y3 in np.linspace(-2, 2, 4):
        draw_connecting_arrow(ax, (x2, y2), radius, (x3, y3), radius)

for y3 in np.linspace(-2, 2, 4):
    for y4 in np.linspace(-2, 2, 4):
        draw_connecting_arrow(ax, (x3, y3), radius, (x4, y4), radius)

for y4 in np.linspace(-2, 2, 4):
    for y5 in np.linspace(-2, 2, 4):
        draw_connecting_arrow(ax, (x4, y4), radius, (x5, y5), radius)

for y5 in np.linspace(-2, 2, 4):
    draw_connecting_arrow(ax, (x5, y5), radius, (x6, 0), radius)

# =============================
# Add text labels
# =============================
plt.text(x1, 4, 'Input Layer', ha='center', va='top', fontsize=14)
plt.text(x2, 4, 'Hidden Layer', ha='center', va='top', fontsize=14)
plt.text(x3, 4, 'Hidden Layer', ha='center', va='top', fontsize=14)
plt.text(x4, 4, 'Hidden Layer', ha='center', va='top', fontsize=14)
plt.text(x5, 4, 'Hidden Layer', ha='center', va='top', fontsize=14)
plt.text(x6, 4, 'Output Layer', ha='center', va='top', fontsize=14)

ax.set_aspect('equal')
plt.xlim(-10, 10)
plt.ylim(-5, 5)
plt.show()



# Modelul folosit de mine
# def my_model_5(width=200, height=200, depth=3, classes=9):
#     inp_img = Input(shape=(width, height, depth))
#
#     model = Conv2D(32, (3, 3), activation='relu')(inp_img)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
#     model = Conv2D(64, (3, 3), activation='relu')(model)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
#     model = Conv2D(128, (3, 3), activation='relu')(model)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
#     model = Conv2D(128, (3, 3), activation='relu')(model)
#     model = MaxPooling2D(pool_size=(2, 2))(model)
#     model = Flatten()(model)
#     model = Dropout(0.5)(model)
#     model = Dense(512, activation='relu')(model)
#     model = Dense(1, activation='sigmoid')(model)
#
#     out = Dense(classes, activation='softmax')(model)
#
#     model = Model(inputs=inp_img, outputs=out)
#     return model

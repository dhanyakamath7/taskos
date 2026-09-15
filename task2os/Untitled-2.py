import os
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation




SIZE = 100
TASKS_PER_FRAME = 30



tf.random.set_seed(42)

matrix_a = tf.random.uniform(
    (SIZE, SIZE),
    minval=1,
    maxval=10,
    dtype=tf.float32
)

matrix_b = tf.random.uniform(
    (SIZE, SIZE),
    minval=1,
    maxval=10,
    dtype=tf.float32
)


a_np = matrix_a.numpy()
b_np = matrix_b.numpy()

result = np.zeros(
    (SIZE, SIZE),
    dtype=np.float32
)




class ExecutionLog:

    def __init__(self):
        self.cells = deque()
        self.lock = threading.Lock()

    def add(self, row, column):

        with self.lock:
            self.cells.append((row, column))


execution_log = ExecutionLog()




class MatrixCellWorker:

    def __init__(self, A, B, output, log):
        self.A = A
        self.B = B
        self.output = output
        self.log = log

    def compute(self, position):

        row, column = position

        
        selected_row = self.A[row, :]

       
        selected_column = self.B[:, column]

       
        cell_value = tf.tensordot(
            selected_row,
            selected_column,
            axes=1
        )

        
        self.output[row, column] = cell_value.numpy()

        
        self.log.add(row, column)

        return position


worker = MatrixCellWorker(
    matrix_a,
    matrix_b,
    result,
    execution_log
)



cell_positions = [
    (row, column)
    for row in range(SIZE)
    for column in range(SIZE)
]


print("\n" + "=" * 55)
print("      100 x 100 MATRIX MULTIPLICATION")
print("      TensorFlow + Python Threads")
print("=" * 55)

print(f"\nMatrix A       : {SIZE} x {SIZE}")
print(f"Matrix B       : {SIZE} x {SIZE}")
print(f"Result Matrix  : {SIZE} x {SIZE}")
print(f"Total cells    : {len(cell_positions)}")

workers = os.cpu_count() or 4

print(f"Worker threads : {workers}")
print("\nCalculating...\n")

start = time.perf_counter()

finished = 0

with ThreadPoolExecutor(
    max_workers=workers
) as pool:

    future_list = [
        pool.submit(
            worker.compute,
            position
        )
        for position in cell_positions
    ]

    for future in as_completed(future_list):

        future.result()
        finished += 1


elapsed = time.perf_counter() - start



expected = tf.matmul(
    matrix_a,
    matrix_b
).numpy()

correct = np.allclose(
    result,
    expected
)


print("Calculation finished.")
print(f"Completed cells : {finished}")
print(f"Execution time  : {elapsed:.4f} seconds")

if correct:
    print("Verification     : SUCCESS")
else:
    print("Verification     : FAILED")




print("\nFirst 3 x 3 values of Matrix C:\n")

print(
    np.round(
        result[:3, :3],
        2
    )
)



completed_order = list(
    execution_log.cells
)

animation_result = np.zeros_like(result)




figure = plt.figure(
    figsize=(17, 7)
)

grid = figure.add_gridspec(
    1,
    3,
    width_ratios=[1, 1, 1]
)

axis_a = figure.add_subplot(grid[0, 0])
axis_b = figure.add_subplot(grid[0, 1])
axis_c = figure.add_subplot(grid[0, 2])



plot_a = axis_a.imshow(
    a_np,
    interpolation="nearest"
)

axis_a.set_title(
    "INPUT MATRIX A",
    fontsize=14,
    fontweight="bold"
)

axis_a.set_xlabel("Column")
axis_a.set_ylabel("Row")



plot_b = axis_b.imshow(
    b_np,
    interpolation="nearest"
)

axis_b.set_title(
    "INPUT MATRIX B",
    fontsize=14,
    fontweight="bold"
)

axis_b.set_xlabel("Column")
axis_b.set_ylabel("Row")



plot_c = axis_c.imshow(
    animation_result,
    interpolation="nearest",
    vmin=0,
    vmax=np.max(result)
)

axis_c.set_title(
    "RESULT MATRIX C",
    fontsize=14,
    fontweight="bold"
)

axis_c.set_xlabel("Column")
axis_c.set_ylabel("Row")



row_line, = axis_a.plot(
    [],
    [],
    linewidth=2
)

column_line, = axis_b.plot(
    [],
    [],
    linewidth=2
)




status_text = figure.text(
    0.5,
    0.025,
    "Preparing animation...",
    ha="center",
    fontsize=12
)



total = len(completed_order)

number_of_frames = (
    total + TASKS_PER_FRAME - 1
) // TASKS_PER_FRAME


def animate(frame):

    first = frame * TASKS_PER_FRAME

    last = min(
        first + TASKS_PER_FRAME,
        total
    )

    current = None

    
    for index in range(first, last):

        row, column = completed_order[index]

        animation_result[row, column] = (
            result[row, column]
        )

        current = (
            row,
            column
        )


    plot_c.set_data(
        animation_result
    )


    if current is not None:

        row, column = current

       
        row_line.set_data(
            [0, SIZE - 1],
            [row, row]
        )

 
        column_line.set_data(
            [column, column],
            [0, SIZE - 1]
        )

   
        axis_c.set_title(
            f"RESULT MATRIX C\n"
            f"C[{row}][{column}]",
            fontsize=13,
            fontweight="bold"
        )


    progress = last / total * 100

    status_text.set_text(
        f"Thread execution: "
        f"{last:,} / {total:,} cells    "
        f"({progress:.1f}%)"
    )


    return (
        plot_a,
        plot_b,
        plot_c,
        row_line,
        column_line,
        status_text
    )



animation = FuncAnimation(
    figure,
    animate,
    frames=number_of_frames,
    interval=80,
    repeat=False,
    blit=False
)

figure.suptitle(
    "100 × 100 Matrix Multiplication "
    "Using Multithreading",
    fontsize=18,
    fontweight="bold"
)

plt.subplots_adjust(
    top=0.82,
    bottom=0.12,
    left=0.04,
    right=0.98,
    wspace=0.25
)


plt.show()

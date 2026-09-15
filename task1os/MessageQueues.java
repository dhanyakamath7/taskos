package javaprojecttask;


class MessageQueues {

    private final int[] data;
    private final int capacity;

    private int writeIndex = 0;
    private int readIndex = 0;
    private int elements = 0;

    MessageQueues(int capacity) {
        this.capacity = capacity;
        data = new int[capacity];
    }

    // Add an item to the queue
    synchronized void addItem(int value) throws InterruptedException {

        while (elements >= capacity) {
            wait();
        }

        data[writeIndex] = value;
        writeIndex = (writeIndex + 1) % capacity;
        elements++;

        System.out.println("Producer -> Added " + value +
                           " | Items in buffer: " + elements);

        notifyAll();
    }

    // Remove an item from the queue
    synchronized int removeItem() throws InterruptedException {

        while (elements == 0) {
            wait();
        }

        int value = data[readIndex];
        readIndex = (readIndex + 1) % capacity;
        elements--;

        System.out.println("Consumer <- Removed " + value +
                           " | Items in buffer: " + elements);

        notifyAll();

        return value;
    }
}


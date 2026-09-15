package javaprojecttask;


class DataProducer extends Thread {

    private final MessageQueues queue;

    DataProducer(MessageQueues queue) {
        this.queue = queue;
    }

    @Override
    public void run() {

        try {
            for (int value = 1; value <= 10; value++) {

                queue.addItem(value);

                Thread.sleep(250);
            }

        } catch (InterruptedException e) {

            Thread.currentThread().interrupt();
            System.out.println("Producer was interrupted.");
        }
    }
}


// Consumer thread

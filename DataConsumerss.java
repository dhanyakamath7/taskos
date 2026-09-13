package javaprojecttask;




class DataConsumers extends Thread {

    private final MessageQueues queue;

    DataConsumers(MessageQueues queue) {
        this.queue = queue;
    }

    @Override
    public void run() {

        try {
            for (int i = 0; i < 10; i++) {

                queue.removeItem();

                Thread.sleep(450);
            }

        } catch (InterruptedException e) {

            Thread.currentThread().interrupt();
            System.out.println("Consumer was interrupted.");
        }
    }
}


// Main class

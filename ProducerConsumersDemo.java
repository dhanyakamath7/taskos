package javaprojecttask;

public class ProducerConsumersDemo {

    public static void main(String[] args) {

        MessageQueues sharedQueue = new MessageQueues(5);

        DataProducer producerThread =
                new DataProducer(sharedQueue);

        DataConsumers consumerThread =
                new DataConsumers(sharedQueue);

        producerThread.start();
        consumerThread.start();

        try {

            producerThread.join();
            consumerThread.join();

        } catch (InterruptedException e) {

            Thread.currentThread().interrupt();
            System.out.println("Main thread was interrupted.");
        }

        System.out.println("\nExecution completed successfully.");
    }
}


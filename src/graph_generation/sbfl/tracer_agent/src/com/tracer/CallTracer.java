package com.tracer;

import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Stack;
import java.util.concurrent.ConcurrentHashMap;

public class CallTracer {
    // (caller, callee) -> count
    private static final Map<String, Long> callEdges = new ConcurrentHashMap<>();
    
    // Thread-local stack to track the call hierarchy
    private static final ThreadLocal<Stack<String>> threadStack = ThreadLocal.withInitial(Stack::new);

    private static String outputFilePath = "dynamic_call_graph.txt";

    public static void setOutputFile(String path) {
        outputFilePath = path;
    }

    public static void logEnter(String className, String methodName) {
        String callee = className + "#" + methodName;
        Stack<String> stack = threadStack.get();
        
        if (!stack.isEmpty()) {
            String caller = stack.peek();
            String edge = caller + " -> " + callee;
            callEdges.merge(edge, 1L, Long::sum);
        }
        
        stack.push(callee);
    }

    public static void logExit() {
        Stack<String> stack = threadStack.get();
        if (!stack.isEmpty()) {
            stack.pop();
        }
    }

    static {
        // Shutdown hook to save data
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try (FileWriter writer = new FileWriter(outputFilePath)) {
                for (Map.Entry<String, Long> entry : callEdges.entrySet()) {
                    writer.write(entry.getKey() + " : " + entry.getValue() + "\n");
                }
                System.out.println("[Tracer] Dynamic call graph saved to " + outputFilePath);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }));
    }
}

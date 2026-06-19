package com.tracer;

import javassist.*;
import java.io.ByteArrayInputStream;
import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.Instrumentation;
import java.security.ProtectionDomain;

public class TracerAgent implements ClassFileTransformer {

    /** Slash-form prefixes of classes to instrument, e.g. ["com/verify/", "com/dummy/"]. */
    private static volatile String[] instrumentPrefixes = new String[]{"org/apache/commons/lang3/", "com/verify/"};

    public static void premain(String agentArgs, Instrumentation inst) {
        if (agentArgs != null && !agentArgs.isEmpty()) {
            try {
                // Agent arg format: "outputFile" or "outputFile;prefix1,prefix2"
                // prefix uses dot-notation, e.g. "com.dummy,org.apache.commons.lang3"
                String outputFile = agentArgs;
                if (agentArgs.contains(";")) {
                    String[] parts = agentArgs.split(";", 2);
                    outputFile = parts[0];
                    String[] dotPrefixes = parts[1].split(",");
                    String[] slashPrefixes = new String[dotPrefixes.length];
                    for (int i = 0; i < dotPrefixes.length; i++) {
                        String p = dotPrefixes[i].trim().replace('.', '/');
                        slashPrefixes[i] = p.endsWith("/") ? p : p + "/";
                    }
                    instrumentPrefixes = slashPrefixes;
                }

                // Use reflection to set output file to avoid early loading
                Class<?> tracerClass = Class.forName("com.tracer.CallTracer");
                java.lang.reflect.Method method = tracerClass.getMethod("setOutputFile", String.class);
                method.invoke(null, outputFile);
            } catch (Exception e) {
                System.err.println("[Tracer] Failed to set output: " + e.getMessage());
            }
        }
        inst.addTransformer(new TracerAgent());
    }

    @Override
    public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
                           ProtectionDomain protectionDomain, byte[] classfileBuffer) {
        
        if (className == null) return null;
        
        // Only instrument classes that match one of the configured prefixes
        boolean shouldInstrument = false;
        for (String prefix : instrumentPrefixes) {
            if (className.startsWith(prefix)) {
                shouldInstrument = true;
                break;
            }
        }
        if (!shouldInstrument) {
            return null;
        }

        try {
            ClassPool cp = new ClassPool(true);
            if (loader != null) {
                cp.appendClassPath(new LoaderClassPath(loader));
            }
            
            String dotClassName = className.replace('/', '.').replace('$', '.');
            CtClass ctClass = cp.makeClass(new ByteArrayInputStream(classfileBuffer));
            
            if (ctClass.isInterface()) {
                return null;
            }

            for (CtBehavior method : ctClass.getDeclaredBehaviors()) {
                if (!method.isEmpty()) {
                    try {
                        StringBuilder sig = new StringBuilder("(");
                        CtClass[] params = method.getParameterTypes();
                        for (int i = 0; i < params.length; i++) {
                            if (i > 0) sig.append(",");
                            sig.append(params[i].getName());
                        }
                        sig.append(")");
                        
                        String methodNameWithSig = method.getName() + sig.toString();
                        method.insertBefore("com.tracer.CallTracer.logEnter(\"" + dotClassName + "\", \"" + methodNameWithSig + "\");");
                        method.insertAfter("com.tracer.CallTracer.logExit();", true);
                    } catch (Exception e) {
                        // Ignore
                    }
                }
            }
            
            byte[] b = ctClass.toBytecode();
            ctClass.detach();
            return b;
        } catch (Exception e) {
            return null;
        }
    }
}

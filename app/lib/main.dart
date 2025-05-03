import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:app/Views/auth.dart';
// import 'package:firebase_core/firebase_core.dart';
// import 'firebase_options.dart';  // Make sure you have this file after running flutterfire configure

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // // Initialize Firebase
  // await Firebase.initializeApp(
  //   options: DefaultFirebaseOptions.currentPlatform,  // You need the generated file from FlutterFire CLI
  // );

  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarBrightness: Brightness.dark,
      systemNavigationBarColor: Colors.transparent,
      systemNavigationBarDividerColor: Colors.transparent,
      systemNavigationBarIconBrightness: Brightness.dark,
      statusBarIconBrightness: Brightness.dark,
    ),
  );

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF009CFF)),
        useMaterial3: true,
      ),
      // Our first screen
      home: const AuthScreen(),
    );
  }
}

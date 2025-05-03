import 'package:flutter/material.dart';
import 'package:app/Components/button.dart';
import 'package:app/Components/colors.dart';
import 'package:app/Components/textfield.dart';
import 'package:app/JSON/users.dart';
import 'package:app/Views/signup.dart';
import 'package:app/Views/home.dart';

import '../SQLite/database_helper.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final usrName = TextEditingController();
  final password = TextEditingController();
  bool isChecked = false;
  bool isLoginTrue = false;
  String errorMessage = "";
  final db = DatabaseHelper();

  login() async {
    if (usrName.text.isEmpty) {
      usrName.text = "Prithak"; 
    }
    if (password.text.isEmpty) {
      password.text = "Bantawa"; 
    }

    if (usrName.text.isEmpty || password.text.isEmpty) {
      setState(() {
        isLoginTrue = true;
        errorMessage = "Both fields are required.";
      });
      return;
    }

    var res = await db.authenticate(Users(usrName: usrName.text, password: password.text));
    
    if (res == true) {
      Users? usrDetails = await db.getUser(usrName.text);

      if (usrDetails != null) {
        if (!mounted) return;
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => HomePage(profile: usrDetails),
          ),
        );
      }
    } else {
      setState(() {
        isLoginTrue = true;
        errorMessage = "Username or password is incorrect.";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: SingleChildScrollView(
          child: SafeArea(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text(
                  "LOGIN",
                  style: TextStyle(color: Color.fromARGB(255, 0, 135, 218), fontSize: 40),
                ),
                Image.asset("assets/background.jpg"),
                InputField(hint: "Username", icon: Icons.account_circle, controller: usrName),
                InputField(hint: "Password", icon: Icons.lock, controller: password, passwordInvisible: true),

                ListTile(
                  horizontalTitleGap: 2,
                  title: const Text("Remember me"),
                  leading: Checkbox(
                    activeColor: primaryColor,
                    value: isChecked,
                    onChanged: (value) {
                      setState(() {
                        isChecked = !isChecked;
                      });
                    },
                  ),
                ),

                Button(label: "LOGIN", press: () {
                  login();
                }),

                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text("Don't have an account?", style: TextStyle(color: Colors.grey)),
                    TextButton(
                      onPressed: () {
                        Navigator.push(context, MaterialPageRoute(builder: (context) => const SignupScreen()));
                      },
                      child: const Text("SIGN UP"),
                    ),
                  ],
                ),

                isLoginTrue
                    ? Text(errorMessage, style: TextStyle(color: Colors.red.shade900))
                    : const SizedBox(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

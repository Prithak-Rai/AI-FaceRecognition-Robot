// import 'package:flutter/material.dart';

// class InputField extends StatelessWidget {
//   final String hint;
//   final IconData icon;
//   final TextEditingController controller;
//   final bool passwordInvisible;
//   final String? errorText;

//   const InputField({
//     required this.hint,
//     required this.icon,
//     required this.controller,
//     this.passwordInvisible = false,
//     this.errorText,
//   });

//   @override
//   Widget build(BuildContext context) {
//     return Padding(
//       padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
//       child: TextField(
//         controller: controller,
//         obscureText: passwordInvisible,
//         decoration: InputDecoration(
//           hintText: hint,
//           prefixIcon: Icon(icon),
//           errorText: errorText,
//           border: OutlineInputBorder(
//             borderRadius: BorderRadius.circular(10),
//           ),
//         ),
//       ),
//     );
//   }
// }

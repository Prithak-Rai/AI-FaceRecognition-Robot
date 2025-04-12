import 'package:flutter/material.dart';
import 'package:flutter_sqlite_auth_app/Components/button.dart';
import 'package:flutter_sqlite_auth_app/Components/textfield.dart';
import 'package:flutter_sqlite_auth_app/JSON/users.dart';
import 'package:flutter_sqlite_auth_app/SQLite/database_helper.dart';

class EditProfileScreen extends StatefulWidget {
  final Users profile;

  const EditProfileScreen({super.key, required this.profile});

  @override
  State<EditProfileScreen> createState() => _EditProfileScreenState();
}

class _EditProfileScreenState extends State<EditProfileScreen> {
  final fullName = TextEditingController();
  final username = TextEditingController();
  final currentPassword = TextEditingController();
  final newPassword = TextEditingController();
  final confirmPassword = TextEditingController();
  final db = DatabaseHelper();

  bool isError = false;
  String errorMessage = "";
  bool isChangingPassword = false;

  @override
  void initState() {
    super.initState();
    // Initialize controllers with current user data
    fullName.text = widget.profile.fullName ?? "";
    username.text = widget.profile.usrName;
  }

  Future<void> updateProfile() async {
    // Reset error state
    setState(() {
      isError = false;
      errorMessage = "";
    });

    // Validate if required fields are empty
    if (fullName.text.isEmpty || username.text.isEmpty) {
      setState(() {
        isError = true;
        errorMessage = "Name and username are required.";
      });
      return;
    }

    // Check if username is changed and already exists
    if (username.text != widget.profile.usrName) {
      bool usernameExists = await db.checkUsernameExists(username.text, widget.profile.usrId);
      if (usernameExists) {
        setState(() {
          isError = true;
          errorMessage = "Username is already taken.";
        });
        return;
      }
    }

    // If changing password, validate password fields
    if (isChangingPassword) {
      // Verify current password
      bool isPasswordCorrect = await db.checkPassword(
        widget.profile.usrName,
        currentPassword.text,
      );

      if (!isPasswordCorrect) {
        setState(() {
          isError = true;
          errorMessage = "Current password is incorrect.";
        });
        return;
      }

      // Check if new passwords match
      if (newPassword.text != confirmPassword.text) {
        setState(() {
          isError = true;
          errorMessage = "New passwords do not match.";
        });
        return;
      }

      // Check if new password is not empty
      if (newPassword.text.isEmpty) {
        setState(() {
          isError = true;
          errorMessage = "New password cannot be empty.";
        });
        return;
      }
    }

    // Create updated user object
    Users updatedUser = Users(
      usrId: widget.profile.usrId,
      fullName: fullName.text,
      email: widget.profile.email,
      usrName: username.text,
      password: isChangingPassword ? newPassword.text : widget.profile.password,
    );

    // Update user in database
    int result = await db.updateUser(updatedUser);

    if (result > 0) {
      // If update successful, return the updated user to previous screen
      if (!mounted) return;
      
      // Show success message
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Profile updated successfully!"),
          backgroundColor: Colors.green,
        ),
      );
      
      // Return to previous screen with updated profile
      Navigator.pop(context, updatedUser);
    } else {
      setState(() {
        isError = true;
        errorMessage = "Failed to update profile. Username or email may already be in use.";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Edit Profile"),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 20),
              const Text(
                "Personal Information",
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 20),
              InputField(
                hint: "Full name", 
                icon: Icons.person, 
                controller: fullName,
              ),
              InputField(
                hint: "Username", 
                icon: Icons.account_circle, 
                controller: username,
              ),
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 16.0),
                child: Divider(),
              ),
              
              // Password Change Section
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    "Change Password",
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Switch(
                    value: isChangingPassword,
                    onChanged: (value) {
                      setState(() {
                        isChangingPassword = value;
                        // Clear password fields when toggling
                        currentPassword.clear();
                        newPassword.clear();
                        confirmPassword.clear();
                      });
                    },
                    activeColor: Colors.blue,
                  ),
                ],
              ),
              
              if (isChangingPassword) ...[
                const SizedBox(height: 16),
                InputField(
                  hint: "Current password", 
                  icon: Icons.lock_outline, 
                  controller: currentPassword,
                  passwordInvisible: true,
                ),
                InputField(
                  hint: "New password", 
                  icon: Icons.lock, 
                  controller: newPassword,
                  passwordInvisible: true,
                ),
                InputField(
                  hint: "Confirm new password", 
                  icon: Icons.lock, 
                  controller: confirmPassword,
                  passwordInvisible: true,
                ),
              ],
              
              const SizedBox(height: 30),
              
              // Display error message if there is one
              if (isError)
                Padding(
                  padding: const EdgeInsets.only(bottom: 16.0),
                  child: Text(
                    errorMessage,
                    style: TextStyle(color: Colors.red.shade900),
                  ),
                ),
              
              // Save Button
              Button(
                label: "SAVE CHANGES", 
                press: updateProfile,
              ),
              
              const SizedBox(height: 16),
              
              // Cancel Button
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: const Center(
                  child: Text(
                    "CANCEL",
                    style: TextStyle(
                      color: Colors.grey,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
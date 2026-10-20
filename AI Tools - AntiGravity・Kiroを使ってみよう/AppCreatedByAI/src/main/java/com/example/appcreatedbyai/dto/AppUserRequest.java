package com.example.appcreatedbyai.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public class AppUserRequest {

    @NotBlank(message = "ユーザー名 (name) は必須です")
    @Size(max = 100, message = "ユーザー名は100文字以内で入力してください")
    private String name;

    @NotBlank(message = "メールアドレス (email) は必須です")
    @Email(message = "有効なメールアドレス形式で入力してください")
    @Size(max = 100, message = "メールアドレスは100文字以内で入力してください")
    private String email;

    public AppUserRequest() {
    }

    public AppUserRequest(String name, String email) {
        this.name = name;
        this.email = email;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }
}

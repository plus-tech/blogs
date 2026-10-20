package com.example.appcreatedbyai.model;

import java.time.LocalDateTime;

public class AppUser {

    private Long id;
    private String name;
    private String email;
    private LocalDateTime createAt;

    public AppUser() {
    }

    public AppUser(Long id, String name, String email, LocalDateTime createAt) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.createAt = createAt;
    }

    public AppUser(String name, String email) {
        this.name = name;
        this.email = email;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
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

    public LocalDateTime getCreateAt() {
        return createAt;
    }

    public void setCreateAt(LocalDateTime createAt) {
        this.createAt = createAt;
    }
}

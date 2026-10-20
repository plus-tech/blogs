package com.example.appcreatedbyai.dto;

import com.example.appcreatedbyai.model.AppUser;
import com.fasterxml.jackson.annotation.JsonFormat;

import java.time.LocalDateTime;

public class AppUserResponse {

    private Long id;
    private String name;
    private String email;

    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime createAt;

    public AppUserResponse() {
    }

    public AppUserResponse(Long id, String name, String email, LocalDateTime createAt) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.createAt = createAt;
    }

    public static AppUserResponse fromModel(AppUser user) {
        return new AppUserResponse(
            user.getId(),
            user.getName(),
            user.getEmail(),
            user.getCreateAt()
        );
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

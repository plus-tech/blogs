package com.example.appcreatedbyai.controller;

import com.example.appcreatedbyai.dto.AppUserRequest;
import com.example.appcreatedbyai.dto.AppUserResponse;
import com.example.appcreatedbyai.service.AppUserService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/app-users")
public class AppUserController {

    private final AppUserService appUserService;

    public AppUserController(AppUserService appUserService) {
        this.appUserService = appUserService;
    }

    /**
     * 全ユーザー取得 (GET /api/app-users)
     */
    @GetMapping
    public ResponseEntity<List<AppUserResponse>> getAllUsers() {
        List<AppUserResponse> users = appUserService.getAllUsers();
        return ResponseEntity.ok(users);
    }

    /**
     * ユーザー1件取得 (GET /api/app-users/{id})
     */
    @GetMapping("/{id}")
    public ResponseEntity<AppUserResponse> getUserById(@PathVariable("id") Long id) {
        AppUserResponse user = appUserService.getUserById(id);
        return ResponseEntity.ok(user);
    }

    /**
     * ユーザー新規登録 (POST /api/app-users)
     */
    @PostMapping
    public ResponseEntity<AppUserResponse> createUser(@Valid @RequestBody AppUserRequest request) {
        AppUserResponse createdUser = appUserService.createUser(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdUser);
    }

    /**
     * ユーザー更新 (PUT /api/app-users/{id})
     */
    @PutMapping("/{id}")
    public ResponseEntity<AppUserResponse> updateUser(
            @PathVariable("id") Long id,
            @Valid @RequestBody AppUserRequest request) {
        AppUserResponse updatedUser = appUserService.updateUser(id, request);
        return ResponseEntity.ok(updatedUser);
    }

    /**
     * ユーザー削除 (DELETE /api/app-users/{id})
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteUser(@PathVariable("id") Long id) {
        appUserService.deleteUser(id);
        return ResponseEntity.noContent().build();
    }
}

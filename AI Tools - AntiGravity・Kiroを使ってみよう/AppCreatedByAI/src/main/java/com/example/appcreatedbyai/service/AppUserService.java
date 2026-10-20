package com.example.appcreatedbyai.service;

import com.example.appcreatedbyai.dto.AppUserRequest;
import com.example.appcreatedbyai.dto.AppUserResponse;
import com.example.appcreatedbyai.exception.DuplicateResourceException;
import com.example.appcreatedbyai.exception.ResourceNotFoundException;
import com.example.appcreatedbyai.model.AppUser;
import com.example.appcreatedbyai.repository.AppUserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@Transactional(readOnly = true)
public class AppUserService {

    private final AppUserRepository appUserRepository;

    public AppUserService(AppUserRepository appUserRepository) {
        this.appUserRepository = appUserRepository;
    }

    /**
     * 全ユーザー取得
     */
    public List<AppUserResponse> getAllUsers() {
        return appUserRepository.findAll().stream()
                .map(AppUserResponse::fromModel)
                .collect(Collectors.toList());
    }

    /**
     * ID指定でユーザー取得
     */
    public AppUserResponse getUserById(Long id) {
        AppUser user = appUserRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("ID [" + id + "] のユーザーが見つかりません"));
        return AppUserResponse.fromModel(user);
    }

    /**
     * ユーザー新規登録 (JDBC経由)
     */
    @Transactional
    public AppUserResponse createUser(AppUserRequest request) {
        if (appUserRepository.existsByName(request.getName())) {
            throw new DuplicateResourceException("ユーザー名 [" + request.getName() + "] は既に登録されています");
        }

        AppUser newUser = new AppUser(request.getName(), request.getEmail());
        AppUser savedUser = appUserRepository.insert(newUser);
        return AppUserResponse.fromModel(savedUser);
    }

    /**
     * ユーザー更新 (JDBC経由)
     */
    @Transactional
    public AppUserResponse updateUser(Long id, AppUserRequest request) {
        AppUser existingUser = appUserRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("ID [" + id + "] のユーザーが見つかりません"));

        if (appUserRepository.existsByNameAndIdNot(request.getName(), id)) {
            throw new DuplicateResourceException("ユーザー名 [" + request.getName() + "] は既に他のユーザーに使用されています");
        }

        existingUser.setName(request.getName());
        existingUser.setEmail(request.getEmail());

        appUserRepository.update(existingUser);
        return AppUserResponse.fromModel(existingUser);
    }

    /**
     * ユーザー削除 (JDBC経由)
     */
    @Transactional
    public void deleteUser(Long id) {
        if (!appUserRepository.findById(id).isPresent()) {
            throw new ResourceNotFoundException("ID [" + id + "] のユーザーが見つかりません");
        }
        appUserRepository.deleteById(id);
    }
}

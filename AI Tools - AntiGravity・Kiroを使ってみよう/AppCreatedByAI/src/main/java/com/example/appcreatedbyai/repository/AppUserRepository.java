package com.example.appcreatedbyai.repository;

import com.example.appcreatedbyai.model.AppUser;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public class AppUserRepository {

    private final JdbcTemplate jdbcTemplate;

    private final RowMapper<AppUser> rowMapper = (rs, rowNum) -> {
        Timestamp ts = rs.getTimestamp("create_at");
        LocalDateTime createAt = (ts != null) ? ts.toLocalDateTime() : null;
        return new AppUser(
                rs.getLong("id"),
                rs.getString("name"),
                rs.getString("email"),
                createAt
        );
    };

    public AppUserRepository(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    /**
     * 全ユーザー取得 (JDBC)
     */
    public List<AppUser> findAll() {
        String sql = "SELECT id, name, email, create_at FROM app_users ORDER BY id ASC";
        return jdbcTemplate.query(sql, rowMapper);
    }

    /**
     * ID指定でユーザー取得 (JDBC)
     */
    public Optional<AppUser> findById(Long id) {
        String sql = "SELECT id, name, email, create_at FROM app_users WHERE id = ?";
        try {
            AppUser user = jdbcTemplate.queryForObject(sql, rowMapper, id);
            return Optional.ofNullable(user);
        } catch (EmptyResultDataAccessException e) {
            return Optional.empty();
        }
    }

    /**
     * 名前による重複チェック (JDBC)
     */
    public boolean existsByName(String name) {
        String sql = "SELECT COUNT(*) FROM app_users WHERE name = ?";
        Integer count = jdbcTemplate.queryForObject(sql, Integer.class, name);
        return count != null && count > 0;
    }

    /**
     * 自ID以外の名前重複チェック (JDBC)
     */
    public boolean existsByNameAndIdNot(String name, Long id) {
        String sql = "SELECT COUNT(*) FROM app_users WHERE name = ? AND id <> ?";
        Integer count = jdbcTemplate.queryForObject(sql, Integer.class, name, id);
        return count != null && count > 0;
    }

    /**
     * ユーザー新規登録 (Oracle Sequence採番 + JDBC)
     */
    public AppUser insert(AppUser user) {
        // Oracle シーケンスから採番
        String seqSql = "SELECT app_users_seq.NEXTVAL FROM dual";
        Long newId = jdbcTemplate.queryForObject(seqSql, Long.class);

        // INSERT 実行
        String insertSql = "INSERT INTO app_users (id, name, email, create_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)";
        jdbcTemplate.update(insertSql, newId, user.getName(), user.getEmail());

        return findById(newId).orElseThrow();
    }

    /**
     * ユーザー情報更新 (JDBC)
     */
    public int update(AppUser user) {
        String sql = "UPDATE app_users SET name = ?, email = ? WHERE id = ?";
        return jdbcTemplate.update(sql, user.getName(), user.getEmail(), user.getId());
    }

    /**
     * ユーザー削除 (JDBC)
     */
    public int deleteById(Long id) {
        String sql = "DELETE FROM app_users WHERE id = ?";
        return jdbcTemplate.update(sql, id);
    }
}

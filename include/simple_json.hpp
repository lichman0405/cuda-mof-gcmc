#pragma once

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace mofgcmc {

struct JsonValue {
    enum Type { J_NULL, J_BOOL, J_NUMBER, J_STRING, J_ARRAY, J_OBJECT };
    Type type = J_NULL;
    bool bval = false;
    double dval = 0.0;
    std::string sval;
    std::vector<JsonValue> arr;
    std::map<std::string, JsonValue> obj;

    bool is_object() const { return type == J_OBJECT; }
    bool is_array()  const { return type == J_ARRAY; }
    bool is_string() const { return type == J_STRING; }
    bool is_number() const { return type == J_NUMBER; }
    bool is_bool()   const { return type == J_BOOL; }

    bool has(const std::string& key) const {
        return type == J_OBJECT && obj.find(key) != obj.end();
    }

    const JsonValue* find(const std::string& key) const {
        if (type != J_OBJECT) return nullptr;
        auto it = obj.find(key);
        return (it == obj.end()) ? nullptr : &it->second;
    }

    const JsonValue& at(const std::string& key) const {
        static const JsonValue empty{};
        const JsonValue* p = find(key);
        return p ? *p : empty;
    }

    std::string get_string(const std::string& key, const std::string& def = "") const {
        const JsonValue* p = find(key);
        return (p && p->is_string()) ? p->sval : def;
    }

    double get_double(const std::string& key, double def = 0.0) const {
        const JsonValue* p = find(key);
        return (p && p->is_number()) ? p->dval : def;
    }

    int get_int(const std::string& key, int def = 0) const {
        const JsonValue* p = find(key);
        return (p && p->is_number()) ? (int)(p->dval + (p->dval >= 0 ? 0.5 : -0.5)) : def;
    }

    bool get_bool(const std::string& key, bool def = false) const {
        const JsonValue* p = find(key);
        if (!p) return def;
        if (p->is_bool()) return p->bval;
        if (p->is_number()) return (p->dval != 0.0);
        return def;
    }
};

namespace detail {

class JsonParser {
public:
    explicit JsonParser(std::string src) : s_(std::move(src)), pos_(0) {}

    bool parse(JsonValue& out, std::string& err) {
        skip_ws();
        if (!parse_value(out, err)) return false;
        skip_ws();
        if (pos_ != s_.size()) {
            err = "Unexpected trailing characters at byte " + std::to_string(pos_);
            return false;
        }
        return true;
    }

private:
    std::string s_;
    size_t pos_;

    void skip_ws() {
        while (pos_ < s_.size() && std::isspace((unsigned char)s_[pos_])) ++pos_;
    }

    bool expect(char c, std::string& err) {
        skip_ws();
        if (pos_ >= s_.size() || s_[pos_] != c) {
            err = std::string("Expected '") + c + "' at byte " + std::to_string(pos_);
            return false;
        }
        ++pos_;
        return true;
    }

    bool parse_value(JsonValue& out, std::string& err) {
        skip_ws();
        if (pos_ >= s_.size()) {
            err = "Unexpected end of JSON input";
            return false;
        }
        char ch = s_[pos_];
        if (ch == '{') return parse_object(out, err);
        if (ch == '[') return parse_array(out, err);
        if (ch == '"') {
            out = JsonValue{};
            out.type = JsonValue::J_STRING;
            return parse_string(out.sval, err);
        }
        if (ch == 't' && s_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            out = JsonValue{};
            out.type = JsonValue::J_BOOL;
            out.bval = true;
            return true;
        }
        if (ch == 'f' && s_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            out = JsonValue{};
            out.type = JsonValue::J_BOOL;
            out.bval = false;
            return true;
        }
        if (ch == 'n' && s_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            out = JsonValue{};
            out.type = JsonValue::J_NULL;
            return true;
        }
        if (ch == '-' || std::isdigit((unsigned char)ch)) {
            return parse_number(out, err);
        }
        err = std::string("Unexpected token '") + ch + "' at byte " + std::to_string(pos_);
        return false;
    }

    bool parse_string(std::string& out, std::string& err) {
        if (!expect('"', err)) return false;
        out.clear();
        while (pos_ < s_.size()) {
            char ch = s_[pos_++];
            if (ch == '"') return true;
            if (ch == '\\') {
                if (pos_ >= s_.size()) { err = "Unterminated escape"; return false; }
                char esc = s_[pos_++];
                switch (esc) {
                    case '"':  out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/':  out.push_back('/'); break;
                    case 'b':  out.push_back('\b'); break;
                    case 'f':  out.push_back('\f'); break;
                    case 'n':  out.push_back('\n'); break;
                    case 'r':  out.push_back('\r'); break;
                    case 't':  out.push_back('\t'); break;
                    default:   out.push_back(esc); break;
                }
            } else {
                out.push_back(ch);
            }
        }
        err = "Unterminated JSON string";
        return false;
    }

    bool parse_number(JsonValue& out, std::string& err) {
        size_t start = pos_;
        if (s_[pos_] == '-') ++pos_;
        while (pos_ < s_.size() && (std::isdigit((unsigned char)s_[pos_]) ||
               s_[pos_] == '.' || s_[pos_] == 'e' || s_[pos_] == 'E' ||
               s_[pos_] == '+' || s_[pos_] == '-')) {
            if ((s_[pos_] == '+' || s_[pos_] == '-') &&
                s_[pos_ - 1] != 'e' && s_[pos_ - 1] != 'E') break;
            ++pos_;
        }
        std::string sub = s_.substr(start, pos_ - start);
        char* endp = nullptr;
        double v = std::strtod(sub.c_str(), &endp);
        if (endp == sub.c_str() || *endp != '\0') {
            err = "Invalid JSON number: " + sub;
            return false;
        }
        out = JsonValue{};
        out.type = JsonValue::J_NUMBER;
        out.dval = v;
        return true;
    }

    bool parse_array(JsonValue& out, std::string& err) {
        if (!expect('[', err)) return false;
        out = JsonValue{};
        out.type = JsonValue::J_ARRAY;
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return true; }
        while (true) {
            JsonValue elem;
            if (!parse_value(elem, err)) return false;
            out.arr.push_back(std::move(elem));
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return true; }
            if (!expect(',', err)) return false;
        }
    }

    bool parse_object(JsonValue& out, std::string& err) {
        if (!expect('{', err)) return false;
        out = JsonValue{};
        out.type = JsonValue::J_OBJECT;
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return true; }
        while (true) {
            skip_ws();
            std::string key;
            if (!parse_string(key, err)) return false;
            if (!expect(':', err)) return false;
            JsonValue val;
            if (!parse_value(val, err)) return false;
            out.obj[std::move(key)] = std::move(val);
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return true; }
            if (!expect(',', err)) return false;
        }
    }
};

} // namespace detail

inline bool parse_json_file(const std::string& path, JsonValue& out, std::string& err) {
    std::ifstream fin(path);
    if (!fin) {
        err = "Cannot open JSON file: " + path;
        return false;
    }
    std::ostringstream ss;
    ss << fin.rdbuf();
    detail::JsonParser p(ss.str());
    return p.parse(out, err);
}

} // namespace mofgcmc

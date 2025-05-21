#ifndef __LOGGER_H__
#define __LOGGER_H__

#ifndef LOGCPP_EXT
#define LOGCPP_EXT 1
#endif

#include <generic/Singleton.h>
// #define DEBUGFLAG true
#if LOGCPP_EXT
#define DEBUGFLAG 1
#else 
#define DEBUGFLAG 0
#endif

#define CHECKS 1

#if DEBUGFLAG
#include <log4cpp/Category.hh>
#include <string>
#endif // DEBUGFLAG
using std::string;

namespace PyTZone {
namespace SingleLog4 {

#if DEBUGFLAG
// #define PATH_TO_WD CppExt/Logger/logs/wd.log
#define PATH_TO_WDLOG "/home/yyh/2.Programs/3.workplace/3.pytorch/2.demo/cppExtension/CppExt/Logger/logs/wd.log"

class logger {
public:
    enum Priority {
        // FATAL = 300,
        ERROR = 300,
        WARN,
        INFO,
        DEBUG
    };
    // static logger * getInstance();
    // static void destroy();

	// class AutoRelease {
    // public:
    //     AutoRelease() {
    //         std::cout << "AutoRelease()" << std::endl;
    //     }
    //     ~AutoRelease() {
    //         std::cout << "~AutoRelease()" << std::endl;
    //         if(_pInstance) {
    //             delete _pInstance;
    //             _pInstance = nullptr;
    //         }
    //     }
    // };

// ===========================================================

	template <class... Args>
	void error(const char * msg, Args... args)
	{
		_cat.error(msg, args...);
	}

	template <class... Args>
	void warn(const char * msg, Args... args)
	{
		_cat.warn(msg, args...);
	}

	template <class... Args>
	void info(const char * msg, Args... args)
	{
		_cat.info(msg, args...);
	}

	template <class... Args>
	void debug(const char * msg, Args... args)
	{
		_cat.debug(msg, args...);
	}

	void error(const char * msg);
	void warn(const char * msg);
	void info(const char * msg);
	void debug(const char * msg);

// ===========================================================

    void setPriority(Priority pri);
public:
    logger();
    ~logger();
private:
    // static logger * _pInstance;
    log4cpp::Category & _cat;
	// static AutoRelease _ar;
};
#endif // DEBUGFLAG

//##__VA_ARGS__ 宏前面加上##的作用在于，当可变参数的个数为0时，
//这里的##起到把前面多余的","去掉的作用,否则会编译出错
/* #define LogError(msg) Mylogger::getInstance()->error(prefix(msg)) */
// 在宏定义中 ... 允许宏接受任意数量的参数
// #define LogError(msg, ...) logger::getInstance()->error(prefix(msg), ##__VA_ARGS__)
// #define LogWarn(msg, ...) logger::getInstance()->warn(prefix(msg), ##__VA_ARGS__)
// #define LogInfo(msg, ...) logger::getInstance()->info(prefix(msg), ##__VA_ARGS__)
// #define LogDebug(msg, ...) logger::getInstance()->debug(prefix(msg), ##__VA_ARGS__)

#if DEBUGFLAG
#define prefix_log(msg) string("[")\
	.append(__FILE__).append(":")\
	.append(__FUNCTION__).append(":")\
	.append(std::to_string(__LINE__)).append("] ")\
	.append(msg).c_str()

#define LogError(msg, ...) Singleton<SingleLog4::logger>::getInstance()->error(prefix_log(msg), ##__VA_ARGS__)
#define LogWarn(msg, ...) Singleton<SingleLog4::logger>::getInstance()->warn(prefix_log(msg), ##__VA_ARGS__)
#define LogInfo(msg, ...) Singleton<SingleLog4::logger>::getInstance()->info(prefix_log(msg), ##__VA_ARGS__)
#define LogDebug(msg, ...) Singleton<SingleLog4::logger>::getInstance()->debug(prefix_log(msg), ##__VA_ARGS__)
#else
#define prefix_log(msg)
#define LogError(msg, ...) 
#define LogWarn(msg, ...) 
#define LogInfo(msg, ...) 
#define LogDebug(msg, ...) 
#endif // DEBUGFLAG

} // namespace end of PyTZone  
} // end of SingleLog4

#endif


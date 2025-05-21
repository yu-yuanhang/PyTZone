#include "logger.h"
#if DEBUGFLAG
#include <log4cpp/PatternLayout.hh>
#include <log4cpp/Priority.hh>
#include <log4cpp/OstreamAppender.hh>
#include <log4cpp/FileAppender.hh>

#include <iostream>

namespace PyTZone {
namespace SingleLog4 {

using std::cout;
using std::endl;

// 静态变量初始化
// logger * logger::_pInstance = nullptr;
// typename logger::AutoRelease logger::_ar;

// logger * logger::getInstance() {
//     if (nullptr == _pInstance) {
//         _pInstance = new logger();
//     }
//     return _pInstance;
// }

// void logger::destroy()
// {
// 	if(_pInstance) 
//     {
// 		delete _pInstance;
//         _pInstance = nullptr;
// 	}
// }


logger::logger()
    :_cat(log4cpp::Category::getRoot().getInstance("cat"))
{

    using namespace log4cpp;
    cout << "logger()" << endl;

    //日志的格式
	PatternLayout * ppl1 = new PatternLayout();
	ppl1->setConversionPattern("%d %c [%p] %m%n");

	PatternLayout * ppl2 = new PatternLayout();
	ppl2->setConversionPattern("%d %c [%p] %m%n");

    //日志的目的地
	OstreamAppender *poa = new OstreamAppender("OstreamAppender", &cout);
	poa->setLayout(ppl1);

    // 设置 path 
	FileAppender *pfa = new FileAppender("FileAppender", PATH_TO_WDLOG);
	pfa->setLayout(ppl2);

    //添加日志目的地到Category
	_cat.addAppender(poa);
	_cat.addAppender(pfa);

    //日志的优先级
	_cat.setPriority(log4cpp::Priority::DEBUG);
}

logger::~logger() {
    cout << "~logger()" << endl;
    log4cpp::Category::shutdown();
}

void logger::error(const char *msg)
{
	_cat.error(msg);
}

void logger::warn(const char *msg)
{
	_cat.warn(msg);
}

void logger::info(const char *msg)
{
	_cat.info(msg);
}

void logger::debug(const char *msg)
{
	_cat.debug(msg);
}

void logger::setPriority(Priority pri)
{
    switch(pri)
	{
	case ERROR:
		_cat.setPriority(log4cpp::Priority::ERROR);
		break;
    case WARN:
		_cat.setPriority(log4cpp::Priority::WARN);
		break;
	case INFO:
		_cat.setPriority(log4cpp::Priority::INFO);
		break;
	case DEBUG:
		_cat.setPriority(log4cpp::Priority::DEBUG);
		break;
	}
}


} // namespace SingleLog4
} // namespace end of PyTZone  

#endif // DEBUGFLAG
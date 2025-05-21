#ifndef __TEMPLATELIST_H__
#define __TEMPLATELIST_H__

#include <iostream>
#include <memory>

using std::cout;
using std::cin;
using std::endl;
using std::unique_ptr;

namespace PyTZone {

struct FILECloser    {
    void operator()(FILE *fp) {
        if (fp) {
            cout << "fclose()" << endl;
            fclose(fp);
        }
    }
};

// 这里是一个链表的类模板 用于在多核模块中方便管理 subNet subLayer 

template <typename T>
class Node {
public:
    Node(): _data(nullptr), _next(nullptr), _pre(nullptr) {}
    Node(T *data): _data(data), _next(nullptr), _pre(nullptr) {}
    ~Node() {
        if (nullptr != _data) delete _data;
        _next = nullptr;
        _pre = nullptr;
    }
    void print() const {cout << *_data << endl;}

public:
    T *_data;   // head
    Node *_next;
    Node *_pre;    
};

// 这里如果 T 是自定义的类型需要重定义输出运算符
template <typename T>
class List {
public:
    List(): _head(new Node<T>()), _tail(new Node<T>()), 
            _ptr(nullptr), _size(0) {
        _head->_next = _tail;
        _tail->_pre = _head;
    }   
    ~List() {
        _ptr = _head->_next;
        while (_ptr != _tail) {
            _head->_next = _ptr->_next;
            _ptr->_next->_pre = _head;
            delete _ptr;
            _ptr = nullptr;
            _ptr = _head->_next;
            _size--;
        }
        delete _head;
        _head = nullptr;
        _ptr = nullptr;
        delete _tail;
        _tail = nullptr;
    }
    size_t size() const {return _size;}

    void push_front(T *data) {
        _ptr = new Node<T>(data);
        _ptr->_next = _head->_next;
        _ptr->_pre = _head;
        _ptr->_next->_pre = _ptr;
        _head->_next = _ptr;
    }
    void push_behind(T *data) {
        _ptr = new Node<T>(data);
        _ptr->_next = _tail;
        _ptr->_pre = _tail->_pre;
        _ptr->_pre->_next = _ptr;
        _tail->_pre = _ptr;
    }

    void delete_front() {
        _ptr = _head->_next;
        if (_ptr == _tail) return;
        _head->_next = _ptr->_next;
        _ptr->_next->_pre = _head;
        delete _ptr;
        _ptr = nullptr;
    }
    void delete_behind() {
        _ptr = _tail->_pre;
        if (_ptr == _head) return;
        _tail->_pre = _ptr->_pre;
        _ptr->_pre->_next = _tail;
        delete _ptr;
        _ptr = nullptr;
    }
    Node<T> *get_ptr() {
        if (_ptr->_next != _tail) _ptr = _ptr->_next;
        return _ptr;    
    }
    void reset_ptr() {_ptr = _head;}

    void display() const {}
    void print() const {
        cout << "List print : size == " << _size << endl;
        Node<T> *ptmp = _head->_next;
        for (int i = 0; i < _size; ++i) {
            cout << "!!!!!!!!!!! idx == 1 !!!!!!!!!!!" << endl;
            ptmp->print();
        }
    }



private:
    Node<T> *_head;
    Node<T> *_tail;
    Node<T> *_ptr;
    size_t _size;
};

}   // end of PyTZone

#endif
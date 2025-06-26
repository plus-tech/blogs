----------------------------------------------------------------
-- Tables and Data for Demonstration
----------------------------------------------------------------

-- Employees DDL
CREATE TABLE Employees (
   Employee_Id NUMBER(6)
 , Employee_Name VARCHAR2(20)
 , Salary NUMBER(8)
 , Manager_Id NUMBER(6)
 , Department_Id NUMBER(4)
)
 NOLOGGING
 PARALLEL
;

INSERT ALL
 INTO Employees VALUES(100, 'David', 4800, null, 10)
 INTO Employees VALUES(101, 'Susan', 6500, 100, 10)
 INTO Employees VALUES(200, 'Jennifer', 4400, null, 20)
 INTO Employees VALUES(201, 'Bruce', 6000, 200, 20)
 INTO Employees VALUES(202, 'Pat', 6000, 200, null)
SELECT * FROM DUAL
;


-- Departments DDL
CREATE TABLE Departments (
   Department_Id NUMBER(4)
 , Department_Name VARCHAR2(20)
 , Manager_Id NUMBER(6)
)
 NOLOGGING
 PARALLEL
;

INSERT INTO Departments
WITH tabrows AS (
 SELECT 10, 'Administration', 100 FROM DUAL UNION ALL
 SELECT 20, 'Marketing', 200 FROM DUAL UNION ALL
 SELECT 99, 'Dummy', null FROM DUAL
)
SELECT * FROM tabrows
;


----------------------------------------------------------------
-- What Joins Can We Have?
----------------------------------------------------------------

-- Inner Join
SELECT * 
FROM Employees emp JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;

-- Left JOin
SELECT * 
FROM Employees emp LEFT JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;

SELECT * 
FROM Employees emp, Departments dpt
WHERE emp.Department_Id = dpt.Department_Id(+)  -- join operator
;

-- Right JOin
SELECT * 
FROM Employees emp RIGHT JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;

SELECT * 
FROM Employees emp, Departments dpt
WHERE emp.Department_Id(+) = dpt.Department_Id  -- join operator
;
 
-- Full Join
SELECT * 
FROM Employees emp FULL JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;

-- Semijoin
SELECT * FROM Employees
 WHERE Department_Id IN 
 (SELECT Department_Id FROM Departments
  )
;

SELECT * FROM Employees emp
 WHERE EXISTS
 (SELECT * FROM Departments dpt
  WHERE emp.Department_Id = dpt.Department_Id
  )
;

-- Antijoin
SELECT * FROM Employees
 WHERE Department_Id NOT IN 
 (SELECT Department_Id FROM Departments
  )
;

SELECT * FROM Employees emp
 WHERE NOT EXISTS
 (SELECT * FROM Departments dpt
  WHERE emp.Department_Id = dpt.Department_Id
  )
;

-- Selfjoin
SELECT * 
FROM Employees emp1, Employees emp2
WHERE emp1.Manager_Id = emp2.Employee_Id
;

-- Cartesian Join
SELECT * 
FROM Employees emp, Departments dpt
;


----------------------------------------------------------------
-- What If We Look from the Perspective of Join Conditions?
----------------------------------------------------------------

-- Band Join
SELECT RPAD(emp1.Employee_Name, 10, ' ') || 
       ' has salary between 100 less and 100 more than ' ||
       RPAD(emp2.Employee_Name, 10, ' ') 
       AS "SALARY COMPARISON"
FROM Employees emp1, Employees emp2
WHERE emp1.Salary BETWEEN emp2.Salary - 100 AND emp2.Salary + 100
;

SELECT RPAD(emp1.Employee_Name, 10, ' ') || 
       ' has salary between 100 less and 100 more than ' ||
       RPAD(emp2.Employee_Name, 10, ' ') 
       AS "SALARY COMPARISON"
FROM Employees emp1, Employees emp2
WHERE emp1.Salary BETWEEN emp2.Salary - 100 AND emp2.Salary + 100
      AND emp1.Employee_Id != emp2.Employee_Id
;


----------------------------------------------------------------
-- A Quick Look at Execution Plan
----------------------------------------------------------------

EXPLAIN PLAN SET STATEMENT_ID = 'JOINS' FOR
SELECT *
FROM Employees emp JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;

SELECT PLAN_TABLE_OUTPUT 
FROM TABLE(DBMS_XPLAN.DISPLAY('PLAN_TABLE', 'JOINS', 'TYPICAL'));


----------------------------------------------------------------
-- Touch Optimizer
----------------------------------------------------------------

SET AUTOTRACE TRACEONLY EXPLAIN

-- Inner Join
SELECT * 
FROM Employees emp JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;

-- Left JOin
SELECT * 
FROM Employees emp LEFT JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;

-- Right JOin
SELECT * 
FROM Employees emp RIGHT JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;


----------------------------------------------------------------
-- Here Come Join Methods
----------------------------------------------------------------

-- Nested Loops Join
CREATE INDEX employees_departments_id_idx ON Employees(Department_Id);

SELECT * 
FROM Employees emp JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
WHERE dpt.Department_Name IN ('Administration', 'Marketing')
;


----------------------------------------------------------------
-- Can We Specify a Join Method?
----------------------------------------------------------------

-- Specify a join method
SELECT /*+ ORDERED USE_NL(dpt) */ 
 emp.Employee_Name, dpt.Department_Name
FROM Employees emp JOIN Departments dpt
ON emp.Department_Id = dpt.Department_Id
;


----------------------------------------------------------------
-- 
----------------------------------------------------------------





CREATE INDEX employees_departments_id_idx ON Employees(Department_Id);